"""Step 7 -- building the final match table.

Scores every candidate pair (not just the labeled sample), keeps those at or
above the threshold, and groups the survivors per Source 1 entity.

Two output shapes, because two consumers need different things:

``entity_matches`` (one row per Source 1 entity)
    ``source1_id, matched_source2_ids, matched_source3_ids`` -- the deliverable
    as specified. Every Source 1 entity appears, including the ones with no
    matches, because "we looked and found nothing" is a meaningful result and
    silently dropping those rows makes the output look complete when it is not.

``pair_scores`` (one row per surviving pair)
    ``source1_id, candidate_id, candidate_source, score`` -- the audit trail.
    Every ID in the entity table can be traced to a score here, and every
    rejection is inspectable. Shipping only the aggregate table makes a wrong
    merge impossible to diagnose.

Multiplicity
-----------
An entity may match zero, one, or many records per source, and many is
legitimate: chains, re-registered entities, and directory duplicates all produce
it. The threshold is therefore applied per pair, and no top-1 restriction is
imposed. Where duplicates are *not* wanted, the right lever is a per-entity
margin (see ``dedupe_by_margin``), not a global threshold.
"""

from __future__ import annotations

import os

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


ID_SEP = "|"


def build_pair_scores(
    features: pd.DataFrame,
    scores: Sequence[float],
    threshold: float,
) -> pd.DataFrame:
    """Attach model scores and split into accepted / rejected."""
    scored = features.copy()
    scored["score"] = np.asarray(scores, dtype=float)
    scored["accepted"] = scored["score"] >= threshold
    return scored


def build_entity_table(
    scored: pd.DataFrame,
    s1: pd.DataFrame,
    s1_id_col: str,
    threshold: float,
) -> pd.DataFrame:
    """One row per Source 1 entity, with matched candidate ids per source.

    Sources with no accepted match get an empty string, not NaN, so the CSV is
    unambiguous and joins cleanly downstream.
    """
    accepted = scored[scored["accepted"]]

    grouped: Dict[str, Dict[str, List[str]]] = {}
    for row in accepted.itertuples():
        s1_id = getattr(row, "s1_id")
        src = getattr(row, "candidate_source")
        cid = str(getattr(row, "candidate_id"))
        grouped.setdefault(s1_id, {}).setdefault(src, []).append(cid)

    sources = sorted(accepted["candidate_source"].unique().tolist())
    records = []
    for s1_id in s1[s1_id_col].tolist():
        rec: Dict[str, object] = {"source1_id": s1_id}
        for src in sources:
            ids = grouped.get(s1_id, {}).get(src, [])
            # Sorted + de-duplicated so the output is stable across runs.
            ids = sorted(set(ids))
            rec[f"matched_{src}_ids"] = ID_SEP.join(ids)
            rec[f"n_matched_{src}"] = len(ids)
        rec["n_matched_total"] = sum(
            rec[f"n_matched_{src}"] for src in sources
        )
        records.append(rec)

    return pd.DataFrame(records)


def dedupe_by_margin(
    scored: pd.DataFrame,
    threshold: float,
    margin: float = 0.15,
    keep_top_per_entity: bool = True,
) -> pd.DataFrame:
    """Optional second pass for the "one Source 1 entity, one match" case.

    When the consumer cannot tolerate multiple matches (deduplication rather
    than enrichment), a higher effective threshold is usually the wrong fix
    because it discards genuine second matches along with the spurious ones. A
    per-entity margin is better: keep the best match for an entity, and keep
    further matches only if they score within ``margin`` of it.

    ``margin=0`` keeps strictly only the single best-scoring match per entity
    per source. Callers that want no filtering at all should not call this
    function -- the pipeline only invokes it when ``margin > 0``.
    """
    keep = []
    accepted = scored[scored["accepted"]]
    for (s1_id, src), grp in accepted.groupby(["s1_id", "candidate_source"]):
        grp = grp.sort_values("score", ascending=False)
        best = float(grp["score"].iloc[0])
        for row in grp.itertuples():
            if row.score >= best - margin:
                keep.append(row.Index)
    out = scored.loc[sorted(keep)].copy() if keep else scored.iloc[0:0].copy()
    out["accepted"] = True
    return out


def summarize(
    entity_table: pd.DataFrame,
    scored: pd.DataFrame,
    s1: pd.DataFrame,
    s1_id_col: str,
) -> Dict[str, object]:
    """Counts for the run manifest."""
    n_entities = len(s1)
    accepted = scored[scored["accepted"]]
    per_source: Dict[str, Dict[str, int]] = {}
    for src, grp in accepted.groupby("candidate_source"):
        per_source[src] = {
            "accepted_pairs": int(len(grp)),
            "distinct_source1_entities_matched": int(grp["s1_id"].nunique()),
            "mean_score": round(float(grp["score"].mean()), 4),
            "min_score": round(float(grp["score"].min()), 4),
        }
    for src in sorted(scored["candidate_source"].unique()):
        per_source.setdefault(src, {
            "accepted_pairs": 0, "distinct_source1_entities_matched": 0,
            "mean_score": None, "min_score": None,
        })

    entities_with_any = int((entity_table["n_matched_total"] > 0).sum())
    return {
        "n_source1_entities": n_entities,
        "n_candidate_pairs_scored": int(len(scored)),
        "n_pairs_accepted": int(len(accepted)),
        "acceptance_rate": round(len(accepted) / max(1, len(scored)), 4),
        "entities_with_zero_matches": int(n_entities - entities_with_any),
        "entities_with_zero_matches_pct": round(
            100.0 * (n_entities - entities_with_any) / max(1, n_entities), 2
        ),
        "per_source": per_source,
    }


# ===========================================================================
# Tiered decision and output generation (Changes 7 and 8)
#
# The shape here is *record-oriented*: one row per Source 2/3 record, because
# that is the grain the fallback pass, the ambiguity check and the review queue
# all work at. The Source 1 entity table is then derived from those rows by
# `build_entity_rollup`, so the two views cannot disagree -- there is one
# decision per record and one aggregation of those decisions, not two pipelines.
# ===========================================================================

#: The four statuses a Source 2/3 record can end up in. Fixed strings, because
#: they are the contract the output CSVs and the summary report are written
#: against.
STATUS_MATCHED = "matched"
STATUS_MANUAL_REVIEW = "manual_review"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNMATCHED = "unmatched"
MATCH_STATUSES = [
    STATUS_MATCHED, STATUS_MANUAL_REVIEW, STATUS_AMBIGUOUS, STATUS_UNMATCHED,
]

#: Statuses a human still has to look at. `ambiguous` is routed into the
#: manual-review file because the only difference between the two is *why* the
#: automatic match was refused, and both need a person.
REVIEW_STATUSES = (STATUS_MANUAL_REVIEW, STATUS_AMBIGUOUS)


@dataclass
class DecisionConfig:
    """Tiers, ambiguity margin and conflict vetoes for the final decision.

    high_threshold
        Tuned by `evaluate.tune_threshold` as the F_0.5 maximizer. At or above
        it a record is auto-matched.

    low_threshold
        Below it a record is not a match at all; between the two is
        `manual_review`. The model is unsure, and F_0.5 says a confident guess
        is worth less than a visible question. Validate that it stays below
        `high_threshold` -- a `low` above `high` would make the middle tier
        unreachable and silently turn every uncertain pair into a match.

    ambiguity_margin
        If the best and second-best Source 1 probabilities for the same record
        are within this margin, the result is `ambiguous` rather than
        auto-applied. Same idea as `dedupe_by_margin`, but applied per Source
        2/3 record at decision time, where the alternative is still visible in
        the output.

    block_on_country_conflict / block_on_location_conflict
        A strong name match is not enough on its own. If both records carry a
        country and they differ, the pair cannot be auto-matched however similar
        the names are: name similarity alone must never win against an active
        conflict. City/state conflicts use `location_conflict_floor` to tell a
        genuine conflict ("Paris" vs "Lyon") from a noisy spelling of the same
        place ("High Point" vs "Highpoint"), which would otherwise bury a lot of
        real matches in the review queue.
    """

    high_threshold: float = 0.85
    low_threshold: float = 0.50
    ambiguity_margin: float = 0.05
    block_on_country_conflict: bool = True
    block_on_location_conflict: bool = True
    location_conflict_floor: float = 0.80
    min_second_best_for_ambiguity: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_threshold <= self.high_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= low <= high <= 1, got "
                f"low={self.low_threshold} high={self.high_threshold}"
            )


def _value(frame: pd.DataFrame, row: int, col: str, default: float = float("nan")) -> float:
    """One cell as a float, with anything unusable becoming ``default``."""
    if col not in frame.columns:
        return default
    v = frame[col].iloc[row]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _is_conflict(frame: pd.DataFrame, pos: int, col: str) -> bool:
    """True when a tri-state exact-match flag means "both present, different".

    ``city_exact`` is 1.0 equal, 0.0 both-present-and-different, NaN when a side
    is missing. Only 0.0 counts as a conflict: NaN means nobody published a
    city, which is missingness rather than disagreement.
    """
    if col not in frame.columns:
        return False
    v = frame[col].iloc[pos]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f == 0.0


def _location_conflict(grp: pd.DataFrame, pos: int, floor: float) -> bool:
    """True when the winning pair's city or state actively disagrees.

    "Actively" means both sides published a value, the values are not equal,
    and they are not merely noisy spellings of each other (similarity below
    ``floor``).
    """
    for flag_col, sim_col in (("city_exact", "city_similarity"),
                              ("state_exact", "state_similarity")):
        if not _is_conflict(grp, pos, flag_col):
            continue
        sim = _value(grp, pos, sim_col, default=1.0)
        if not (sim == sim) or sim < floor:
            return True
    return False


def decide_match_status(
    pairs: pd.DataFrame,
    cfg: DecisionConfig,
) -> pd.DataFrame:
    """Collapse scored pairs into one decision row per Source 2/3 record.

    ``pairs`` must carry ``s1_entity_id``, ``source_record_id``,
    ``classifier_probability`` and the feature columns the vetoes read.

    For every record the best-scoring Source 1 entity wins and the tier comes
    from its probability. Two overrides can demote an auto-match:

    * **ambiguous** -- the runner-up Source 1 entity is within
      ``ambiguity_margin`` of the winner. Two Source 1 records explain the same
      Source 2 record about equally well, so neither is applied automatically.
    * **conflict veto** -- the winning pair's country (or city/state) actively
      conflicts. The status becomes `manual_review`, not `unmatched`: the model
      liked the pair, something structural disagrees, and a human is the right
      resolver.

    Returns one row per record with the winning/second-best ids and
    probabilities, the status, and a `decision_notes` string naming any
    override that fired.
    """
    out_cols = [
        "source_record_id", "source_dataset", "matched_source1_id",
        "classifier_probability", "weighted_composite_score",
        "second_best_source1_id", "second_best_probability",
        "match_status", "decision_notes", "n_scored_candidates",
    ]
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame(columns=out_cols)

    # Highest probability first, so position 0 of each group is the winner and
    # the first row with a *different* s1_entity_id is the runner-up.
    ordered = pairs.sort_values(
        ["source_record_id", "classifier_probability"],
        ascending=[True, False], kind="stable",
    )
    rows = []
    for record_id, grp in ordered.groupby("source_record_id", sort=False):
        best_pos = 0
        best_p = float(grp["classifier_probability"].iloc[0])
        # The runner-up must be a *different* Source 1 entity. A repeat of the
        # same entity is a duplicate pair, not a competing explanation.
        others = grp[grp["s1_entity_id"] != grp["s1_entity_id"].iloc[0]]
        second_p = (
            float(others["classifier_probability"].iloc[0]) if len(others)
            else float("nan")
        )

        notes: List[str] = []
        if best_p >= cfg.high_threshold:
            status = STATUS_MATCHED
        elif best_p >= cfg.low_threshold:
            status = STATUS_MANUAL_REVIEW
        else:
            status = STATUS_UNMATCHED

        # --- ambiguous override --------------------------------------------
        if status == STATUS_MATCHED and len(others):
            gap = best_p - second_p
            if gap <= cfg.ambiguity_margin and second_p >= cfg.min_second_best_for_ambiguity:
                status = STATUS_AMBIGUOUS
                notes.append(
                    f"ambiguous:second_best_within_margin({gap:.4f}<={cfg.ambiguity_margin})"
                )

        # --- active-conflict veto ------------------------------------------
        if status == STATUS_MATCHED:
            if cfg.block_on_country_conflict and _is_conflict(grp, best_pos, "country_exact"):
                status = STATUS_MANUAL_REVIEW
                notes.append("veto:country_conflict")
            elif cfg.block_on_location_conflict and _location_conflict(
                grp, best_pos, cfg.location_conflict_floor
            ):
                status = STATUS_MANUAL_REVIEW
                notes.append("veto:city_or_state_conflict")

        rows.append({
            "source_record_id": record_id,
            "source_dataset": grp["source_dataset"].iloc[0]
            if "source_dataset" in grp.columns else "",
            "matched_source1_id": grp["s1_entity_id"].iloc[0],
            "classifier_probability": best_p,
            "weighted_composite_score": _value(grp, best_pos, "weighted_composite_score"),
            "second_best_source1_id": (
                others["s1_entity_id"].iloc[0] if len(others) else ""
            ),
            "second_best_probability": second_p,
            "match_status": status,
            "decision_notes": ";".join(notes),
            "n_scored_candidates": int(len(grp)),
        })
    return pd.DataFrame(rows, columns=out_cols)


#: Column order of the record-level outputs. Identifiers first keeps the file
#: usable with ``cut -f1,3`` during a review session.
RECORD_OUTPUT_COLUMNS: List[str] = [
    "source_record_id", "source_dataset", "matched_source1_id",
    "classifier_probability", "weighted_composite_score", "match_status",
    "candidate_count", "candidate_generation_reasons", "name_similarity",
    "address_similarity", "city_similarity", "city_exact", "state_similarity",
    "state_exact", "country_exact", "phonetic_name_match", "wcs_name",
    "wcs_address", "wcs_city_state", "wcs_country", "second_best_source1_id",
    "second_best_probability", "processing_notes", "cand_business_name_raw",
    "cand_business_address_raw", "cand_country_raw", "s1_business_name_raw",
    "s1_business_address_raw", "s1_country_raw",
]


def build_record_table(
    decisions: pd.DataFrame,
    records: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """One row per Source 2/3 record: the Change 8 output contract.

    Joins three things computed at different grains:

    * ``decisions`` -- the tiered outcome and the winning/second-best pairs.
    * ``records``   -- candidate counts, generation reasons and generation
      notes, including the rows that produced *zero* candidates.
    * ``pairs``     -- the feature values of the winning pair, so a reviewer
      sees the evidence rather than a bare probability.

    A record that never reached the decision layer still gets a row, with
    ``match_status = unmatched`` and the reason in `processing_notes`. That row
    is the visible proof that no real record was silently dropped.
    """
    keep_records = [c for c in (
        "source_record_id", "source_dataset", "candidate_count",
        "candidate_generation_reasons", "generation_status", "processing_notes",
    ) if records is not None and c in records.columns]

    if decisions is None or len(decisions) == 0:
        out = pd.DataFrame(columns=["source_record_id", "match_status",
                                    "decision_notes", "source_dataset"])
    else:
        out = decisions.copy()

    # Winning pair's features, for the manual-review view.
    feature_cols = [
        c for c in (
            "name_similarity", "name_token_set_ratio", "name_token_sort_ratio",
            "name_jaro_winkler", "name_token_overlap_ratio", "address_similarity",
            "city_similarity", "city_exact", "state_similarity", "state_exact",
            "country_exact", "phonetic_name_match", "wcs_name", "wcs_address",
            "wcs_city_state", "wcs_country",
        )
        if c in pairs.columns
    ]
    raw_cols = [c for c in pairs.columns if c.endswith("_raw")]
    winner: Optional[pd.DataFrame] = None
    if pairs is not None and len(pairs):
        winner = (
            pairs.sort_values("classifier_probability", ascending=False, kind="stable")
            .drop_duplicates("source_record_id", keep="first")
            .set_index("source_record_id")
        )

    # Records that never reached the decision layer still belong in the output.
    if records is not None and len(records):
        seen = set(out["source_record_id"]) if "source_record_id" in out else set()
        missing = records[~records["source_record_id"].isin(seen)]
        if len(missing):
            extra = pd.DataFrame({
                "source_record_id": missing["source_record_id"],
                "match_status": STATUS_UNMATCHED,
                "decision_notes": "no_candidate_pairs",
            })
            if "source_dataset" in missing:
                extra["source_dataset"] = missing["source_dataset"].to_numpy()
            for col in out.columns:
                if col not in extra.columns:
                    extra[col] = np.nan
            out = pd.concat([out, extra[out.columns]], ignore_index=True)

    if keep_records:
        gen = records[keep_records].rename(columns={"processing_notes": "generation_notes"})
        gen = gen.drop(columns=[c for c in ("source_dataset",) if c in gen.columns
                                and c in out.columns and c != "source_record_id"])
        out = out.merge(gen, on="source_record_id", how="left")

    if winner is not None and len(out):
        for col in feature_cols + raw_cols:
            if col not in out.columns:
                out[col] = out["source_record_id"].map(winner[col])

    # One note column: what happened in generation, then what happened in the
    # decision, then the generation status. Built with vectorized string ops
    # rather than chained `str.cat`, because `str.cat` yields NaN for every
    # position once the accumulated value is NaN, which silently swallowed
    # every decision note.
    def _note_series(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series("", index=out.index, dtype=object)
        return out[col].fillna("").astype(str).str.strip()

    combined = pd.Series("", index=out.index, dtype=object)
    for col in ("processing_notes", "generation_notes", "decision_notes",
                "generation_status"):
        part = _note_series(col)
        if not part.any():
            continue
        add = (combined == "") & (part != "")
        join = (combined != "") & (part != "")
        combined = combined.mask(join, combined + ";" + part).mask(add, part)
    out["processing_notes"] = combined.replace("", "ok")
    if "candidate_count" in out.columns:
        out["candidate_count"] = pd.to_numeric(
            out["candidate_count"], errors="coerce").fillna(0).astype(int)
    if "candidate_generation_reasons" in out.columns:
        out["candidate_generation_reasons"] = out[
            "candidate_generation_reasons"].fillna("none").replace("", "none")
    if "match_status" in out.columns:
        out["match_status"] = out["match_status"].fillna(STATUS_UNMATCHED)
    return out.sort_values(
        ["source_dataset", "source_record_id"], kind="stable", na_position="first"
    ).reset_index(drop=True)


def build_entity_rollup(
    matched_by_entity: Dict[str, Set[str]],
    cand_by_entity: Dict[str, Set[str]],
    all_s1_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-record decisions into the two challenge submission files.

    ``matching_results.tsv`` -- one row per Source 1 entity listing the Source
    2/3 ids auto-matched to it. **Every** Source 1 entity in ``all_s1_ids``
    appears, including entities that were never a candidate for anything: the
    leaderboard scores singletons, so a missing row changes the denominator
    rather than merely losing a match.

    ``candidate_pairs.tsv`` -- one row per Source 1 entity listing every
    candidate generated for it, which is what a reviewer needs to see the
    blocking recall ceiling. Final matches are always a subset of it.

    A Source 2/3 record is assigned to at most one Source 1 entity (its
    winner), so no id can appear under two entities. That is a deliberate
    consequence of the record-oriented decision: a Source 2 record that fits two
    Source 1 entities equally well is reported `ambiguous` and assigned to
    neither, rather than being merged into both.

    Takes the two entity -> id-set maps rather than the pair frame, because the
    orchestrator builds them chunk by chunk and a 2.2M-entity rollup should not
    require holding 30M pair rows in memory at once.
    """
    results, candidates = [], []
    for s1_id in all_s1_ids:
        key = str(s1_id)
        results.append({
            "source1_entity_id": key,
            "matched_entity_ids": ",".join(sorted(matched_by_entity.get(key, ()))),
        })
        candidates.append({
            "source1_entity_id": key,
            "candidate_entity_ids": ",".join(sorted(cand_by_entity.get(key, ()))),
        })
    results = pd.DataFrame(results, columns=["source1_entity_id", "matched_entity_ids"])
    candidates = pd.DataFrame(
        candidates, columns=["source1_entity_id", "candidate_entity_ids"]
    )

    # Fail loudly rather than submit a short file. `all_s1_ids` is the full
    # Source 1 id list, so a count mismatch means the id list itself was
    # truncated upstream (e.g. a row cap left in place), and every missing row
    # silently changes the denominator of the macro F_0.5 instead of just
    # losing a match.
    _assert_full_coverage(results, candidates, all_s1_ids)
    return results, candidates


def _assert_full_coverage(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    all_s1_ids: Sequence[str],
) -> None:
    """Every Source 1 entity must appear exactly once in both files.

    Asserting rather than warning is deliberate: a short file is a rejection, and
    a rejected submission costs the whole run.
    """
    expected = len(all_s1_ids)
    for name, frame in (("matching_results", results), ("candidate_pairs", candidates)):
        if len(frame) != expected:
            raise AssertionError(
                f"{name}: {len(frame):,} rows but Source 1 has {expected:,} entities "
                f"({expected - len(frame):,} missing). The Source 1 id list is "
                f"truncated -- check for a row cap still set on the source."
            )
        if frame["source1_entity_id"].duplicated().any():
            dupes = frame.loc[
                frame["source1_entity_id"].duplicated(), "source1_entity_id"
            ].head(3).tolist()
            raise AssertionError(
                f"{name}: duplicate source1_entity_id rows, e.g. {dupes}. "
                f"Each Source 1 entity must appear exactly once."
            )
        # Zero matches must be an empty string, never "None"/"[]"/"nan", which
        # is what a naive fill would leave behind.
        blank = frame["matched_entity_ids"] if "matched_entity_ids" in frame else None
        if blank is not None:
            bad = frame[blank.isin(["None", "nan", "[]", "null"])]
            if len(bad):
                raise AssertionError(
                    f"{name}: {len(bad)} rows have a placeholder instead of an empty "
                    f"id list, e.g. {bad['matched_entity_ids'].iloc[0]!r}. "
                    f"Zero-match entities must use an empty string."
                )


def _selfcheck_tsv(path: str, n_expected: int, header: Sequence[str]) -> None:
    """Re-read a written .tsv and confirm it is genuinely tab-separated.

    Guards the failure mode where a comma-separated file is written with a .tsv
    extension: the header then looks plausible but every row is one field, and
    the validator rejects it.
    """
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline().rstrip("\n")
        n_rows = 1
        for _ in fh:
            n_rows += 1
    fields = first.split("\t")
    if len(fields) != 2:
        raise AssertionError(
            f"{os.path.basename(path)}: header splits into {len(fields)} tab-separated "
            f"fields, expected 2. Got {first!r}. A comma-separated file written with a "
            f".tsv extension looks like this."
        )
    if fields != list(header):
        raise AssertionError(
            f"{os.path.basename(path)}: header is {fields}, expected {list(header)}"
        )
    if n_rows != n_expected:
        raise AssertionError(
            f"{os.path.basename(path)}: wrote {n_rows:,} lines but expected "
            f"{n_expected:,} (1 header + {n_expected - 1:,} entities)."
        )


class RecordAccumulator:
    """Streaming writer and counter for the record-level outputs.

    The pipeline decides one Source 2/3 chunk at a time, and the record-level
    output is one row per Source 2/3 record -- 10M rows on the full dataset.
    Holding them all to sort and write at the end is exactly the kind of
    memory blow-up the chunked design exists to avoid, so each chunk is routed
    to its destination file immediately and only bounded state is kept:

    * the two per-entity id sets for the submission rollup (with a hard cap, see
      `max_rollup_pairs`);
    * running counters for the summary report.
    """

    def __init__(
        self,
        out_dir: str,
        max_rollup_pairs: int = 40_000_000,
        write_submission_files: bool = True,
    ) -> None:
        import os

        self.out_dir = out_dir
        self.write_submission_files = write_submission_files
        self.max_rollup_pairs = int(max_rollup_pairs)
        os.makedirs(out_dir, exist_ok=True)
        self.matched_by_entity: Dict[str, Set[str]] = defaultdict(set)
        self.cand_by_entity: Dict[str, Set[str]] = defaultdict(set)
        self.rollup_saturated = False
        self._files: Dict[str, object] = {}
        self._headers: Dict[str, List[str]] = {}
        self.counters: Dict[str, object] = {
            "n_records": 0,
            "n_pairs_scored": 0,
            "counts_per_status": {s: 0 for s in MATCH_STATUSES},
            "per_source": {},
            "sum_candidate_count": 0,
            "n_zero_candidate_records": 0,
            "n_matched": 0,
            "sum_prob_matched": 0.0,
            "n_ambiguous_overridden": 0,
            "n_conflict_vetoed": 0,
            "rule_attribution": {},
            "prob_hist": np.zeros(20, dtype=np.int64),
            "prob_hist_edges": np.linspace(-1.0, 1.0, 21),
        }
        for name in ("matched_records.csv", "manual_review_records.csv",
                     "unmatched_records.csv", "unresolved_records.csv"):
            self._files[name] = open(os.path.join(out_dir, name), "w", newline="")

    # -- writing ------------------------------------------------------------
    def _append(self, name: str, frame: pd.DataFrame) -> None:
        if not len(frame):
            return
        cols = self._headers.get(name)
        frame.to_csv(
            self._files[name],
            index=False,
            header=cols is None,
            lineterminator="\n",
        )
        self._headers[name] = list(frame.columns)

    def add(self, record_table: pd.DataFrame, pairs: Optional[pd.DataFrame] = None) -> None:
        """Route one chunk's decided records to their output files."""
        if record_table is None or not len(record_table):
            return
        cols = [c for c in RECORD_OUTPUT_COLUMNS if c in record_table.columns]
        extra = [c for c in record_table.columns if c not in cols]
        ordered = record_table[cols + extra]

        status = record_table["match_status"]
        self._append("matched_records.csv", ordered[status == STATUS_MATCHED])
        self._append("manual_review_records.csv", ordered[status.isin(REVIEW_STATUSES)])
        self._append("unmatched_records.csv", ordered[status == STATUS_UNMATCHED])
        if "candidate_count" in record_table.columns:
            counts = pd.to_numeric(record_table["candidate_count"], errors="coerce")
            self._append("unresolved_records.csv", ordered[counts.fillna(0) == 0])

        c = self.counters
        c["n_records"] = int(c["n_records"]) + len(record_table)
        c["n_pairs_scored"] = int(c["n_pairs_scored"]) + int(
            record_table["n_scored_candidates"].sum()
            if "n_scored_candidates" in record_table.columns else 0
        )
        for s, n in record_table["match_status"].value_counts().items():
            c["counts_per_status"][s] = int(c["counts_per_status"].get(s, 0)) + int(n)
        for src, grp in record_table.groupby("source_dataset", dropna=False):
            bucket = c["per_source"].setdefault(str(src), {
                "n_records": 0, "counts_per_status": {s: 0 for s in MATCH_STATUSES},
            })
            bucket["n_records"] += len(grp)
            for s, n in grp["match_status"].value_counts().items():
                bucket["counts_per_status"][s] = int(
                    bucket["counts_per_status"].get(s, 0)) + int(n)
        if "candidate_count" in record_table.columns:
            counts = pd.to_numeric(record_table["candidate_count"], errors="coerce")
            c["sum_candidate_count"] = float(c["sum_candidate_count"]) + float(
                counts.fillna(0).sum()
            )
            c["n_zero_candidate_records"] = int(c["n_zero_candidate_records"]) + int(
                (counts.fillna(0) == 0).sum()
            )
        matched = record_table[status == STATUS_MATCHED]
        c["n_matched"] = int(c["n_matched"]) + len(matched)
        if len(matched):
            c["sum_prob_matched"] = float(c["sum_prob_matched"]) + float(
                pd.to_numeric(matched["classifier_probability"], errors="coerce").sum()
            )
        notes = record_table.get("processing_notes", pd.Series(dtype=object)).astype(str)
        c["n_ambiguous_overridden"] = int(c["n_ambiguous_overridden"]) + int(
            notes.str.contains("ambiguous:").sum()
        )
        c["n_conflict_vetoed"] = int(c["n_conflict_vetoed"]) + int(
            notes.str.contains("veto:").sum()
        )
        probs = pd.to_numeric(record_table.get("classifier_probability"), errors="coerce")
        if probs is not None and probs.notna().any():
            hist, _ = np.histogram(
                probs.fillna(-1.0).to_numpy(), bins=20, range=(-1.0, 1.0)
            )
            c["prob_hist"] = np.asarray(c["prob_hist"]) + hist

        # -- per-rule attribution, from the winning pair of each matched record
        if pairs is not None and len(pairs) and "matched_rules" in pairs.columns:
            att = c["rule_attribution"]
            counts = pairs["matched_rules"].astype(str).str.split("|").explode().value_counts()
            for rule, n in counts.items():
                att.setdefault(str(rule), {"candidate_pairs": 0, "matched": 0})
                att[str(rule)]["candidate_pairs"] += int(n)
            if len(matched):
                win = (
                    pairs.sort_values("classifier_probability", ascending=False,
                                      kind="stable")
                    .drop_duplicates("source_record_id", keep="first")
                    .set_index("source_record_id")["matched_rules"]
                )
                matched_ids = set(matched["source_record_id"])
                for record_id in matched_ids:
                    for rule in str(win.get(record_id, "")).split("|"):
                        if rule:
                            att.setdefault(rule, {"candidate_pairs": 0, "matched": 0})
                            att[rule]["matched"] += 1

        # -- rollup state, bounded -----------------------------------------
        if pairs is not None and len(pairs):
            for s1_id, rec_id in zip(pairs["s1_entity_id"], pairs["source_record_id"]):
                if self.rollup_saturated:
                    break
                bucket = self.cand_by_entity[str(s1_id)]
                if rec_id not in bucket:
                    if len(bucket) >= self.max_rollup_pairs:
                        self.rollup_saturated = True
                        break
                    bucket.add(str(rec_id))
        for s1_id, rec_id in zip(
            matched["matched_source1_id"], matched["source_record_id"]
        ):
            self.matched_by_entity.setdefault(str(s1_id), set()).add(str(rec_id))

    def close(self) -> Dict[str, object]:
        for fh in self._files.values():
            fh.close()
        self._files = {}
        return self.counters


def summarize_er_run(
    counters: Dict[str, object],
    block_stats: Dict[str, object],
    fallback_stats: Dict[str, int],
    config_snapshot: Dict[str, object],
    validation: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build the body of ``summary_report.json`` from the streaming counters.

    The only evaluation metric in here is F_0.5, passed in as ``validation``.
    Everything else -- per-status counts, match rate, average candidate count,
    per-rule attribution, unresolved-after-fallback, score distribution -- is
    pipeline health, not model quality, and it is what tells you whether a bad
    F_0.5 is a blocking problem or a modelling problem.
    """
    c = counters
    n = int(c.get("n_records", 0))
    counts = dict(c.get("counts_per_status", {}))  # type: ignore[arg-type]
    per_source = {}
    for src, bucket in dict(c.get("per_source", {})).items():  # type: ignore[arg-type]
        n_src = int(bucket.get("n_records", 0))  # type: ignore[union-attr]
        src_counts = dict(bucket.get("counts_per_status", {}))  # type: ignore[union-attr]
        per_source[str(src)] = {
            "n_records": n_src,
            "counts_per_status": {s: int(src_counts.get(s, 0)) for s in MATCH_STATUSES},
            "match_rate_pct": round(
                100.0 * int(src_counts.get(STATUS_MATCHED, 0)) / max(1, n_src), 2
            ),
        }
    n_matched = int(c.get("n_matched", 0))
    edges = np.asarray(c.get("prob_hist_edges", np.linspace(-1.0, 1.0, 21)), dtype=float)

    report: Dict[str, object] = {
        "n_source2_3_records": n,
        "counts_per_status": {s: int(counts.get(s, 0)) for s in MATCH_STATUSES},
        "match_rate_pct": round(
            100.0 * int(counts.get(STATUS_MATCHED, 0)) / max(1, n), 2
        ),
        "avg_candidate_count": round(
            float(c.get("sum_candidate_count", 0.0)) / max(1, n), 4
        ),
        "avg_classifier_probability_for_matches": (
            round(float(c.get("sum_prob_matched", 0.0)) / n_matched, 4) if n_matched else None
        ),
        "n_candidate_pairs_scored": int(c.get("n_pairs_scored", 0)),
        "n_unresolved_after_fallback": int(c.get("n_zero_candidate_records", 0)),
        "n_ambiguous_overridden": int(c.get("n_ambiguous_overridden", 0)),
        "n_conflict_vetoed": int(c.get("n_conflict_vetoed", 0)),
        "blocking": block_stats,
        "fallback": fallback_stats,
        "match_count_per_blocking_rule": c.get("rule_attribution", {}),
        "per_source": per_source,
        "classifier_probability_distribution": {
            "bins": [round(float(e), 3) for e in edges],
            "counts": [int(x) for x in np.asarray(c.get("prob_hist", []))],
        },
        "config": config_snapshot,
        "reported_metric": "F_0.5 (the only reported evaluation metric)",
    }
    if validation is not None:
        report["validation"] = validation
    return report


def write_er_outputs(
    accumulator: "RecordAccumulator",
    debug_log: pd.DataFrame,
    all_s1_ids: Sequence[str],
    out_dir: str,
    summary: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """Close the streaming writer and write the remaining output files.

    Per-status record files are written incrementally by `RecordAccumulator`;
    what is left for the caller to write is the debug log, the summary report
    and the two challenge submission files:

    ``matching_results.tsv``  one row per Source 1 entity
    ``candidate_pairs.tsv``   the candidate set per Source 1 entity
    ``candidate_debug_log.csv``  oversized / truncated / unresolved diagnostics
    ``summary_report.json``   the counters and F_0.5
    """
    import json
    import os

    accumulator.close()
    paths: Dict[str, str] = {}
    for name in ("matched_records.csv", "manual_review_records.csv",
                 "unmatched_records.csv", "unresolved_records.csv"):
        paths[name] = os.path.join(out_dir, name)

    debug_path = os.path.join(out_dir, "candidate_debug_log.csv")
    debug_log.to_csv(debug_path, index=False)
    paths["candidate_debug_log.csv"] = debug_path

    if accumulator.write_submission_files:
        results, candidates = build_entity_rollup(
            accumulator.matched_by_entity, accumulator.cand_by_entity, all_s1_ids
        )
        for frame, name, header in (
            (results, "matching_results.tsv",
             ("source1_entity_id", "matched_entity_ids")),
            (candidates, "candidate_pairs.tsv",
             ("source1_entity_id", "candidate_entity_ids")),
        ):
            path = os.path.join(out_dir, name)
            # sep="\t" and a .tsv extension. QUOTE_MINIMAL only quotes a field
            # that contains the separator, so the comma-joined id lists are
            # written unquoted -- which is what the validator expects.
            frame.to_csv(path, sep="\t", index=False)
            _selfcheck_tsv(path, len(all_s1_ids) + 1, header)
            paths[name] = path

    if summary is not None:
        path = os.path.join(out_dir, "summary_report.json")

        def _json_default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return None if o != o else float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (set, tuple)):
                return sorted(o)
            return str(o)

        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)
        paths["summary_report.json"] = path
    return paths
