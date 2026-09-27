"""Step 2 -- blocking / candidate generation.

Comparing every Source 1 record against every Source 2/3 record is O(n*m) and
for 1.2k x 3k that is already 3.6M pairs, most of them trivially non-matching.
Blocking collapses that by only comparing records that agree on at least one
cheap key.

Which keys, and why
-------------------
No single key has both high recall and a small block. So we take the *union* of
several independent keys, each of which fails in a different place:

  phone10        Near-perfect precision. A shared 10-digit phone is almost always
                 the same business. Misses everything with a missing phone or a
                 switchboard (30% of Source 2 in the generator), so it cannot
                 stand alone.

  zip5           Strong locality. Misses PO boxes recorded without a zip and
                 anything where one source dropped the zip.

  zip3 + name     zip3 relaxes the zip but re-anchors on a name fragment, so a
                 3-digit zip collision does not create a giant block. This is the
                 workhorse key for records that keep geography but not the exact
                 zip.

  soundex(name)   Phonetic, so it survives small spelling errors ("Hendersen" /
                 "Henderson"). Fails on genuinely different wording
                 ("Bicycle" / "Bike") and on names sharing a soundex.

  name_prefix4   First 4 characters of the name core. Cheap, and a last-resort
                 key for records that kept a name but lost all locality.

  geo_cell       Rounded lat/lon cell, for records whose address is mangled but
                 which are still geocoded. Only available where a source carries
                 coordinates (Source 2 has none, so it contributes nothing
                 there -- which is exactly why the other keys exist).

  name_token     Inverted index over individual informative name tokens. The
                 highest-recall key for the "same brand, different suffix" case
                 ("Acme Bicycle" / "Acme Bike"). Only tokens that are rare
                 enough to be informative are used; a token appearing in 1% of
                 records is a category word, not an identifier.

Union of keys means a pair is generated if ANY key fires. The cost is that a
pair can be reached several times, so we keep a bitmask of which keys produced
it and hand that to the model as features -- a pair that agrees on four
independent keys is not the same evidence as one that agrees on a single
loose key, and the model should be able to see the difference.

Blocking recall
---------------
`blocking_recall` answers the question that actually matters: did blocking
throw away any true match? Every recall number in this pipeline is conditioned
on that check, because a recall of 0.99 on 3% of the true pairs still loses
hundreds of entities. The synthetic run reports recall against the hidden truth
file; a real run reports it against whatever seed pairs you supply.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

try:  # pragma: no cover - optional dependency
    from jellyfish import soundex as _soundex

    def soundex(text: str) -> str:
        if not text:
            return ""
        return str(_soundex(text))
except Exception:  # pragma: no cover
    _SOUDEX_MAP = {
        **dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"), **dict.fromkeys("l", "4"),
        **dict.fromkeys("mn", "5"), **dict.fromkeys("r", "6"),
    }

    def soundex(text: str) -> str:
        """Classic 4-character Soundex, used when jellyfish is unavailable."""
        if not text:
            return ""
        codes = [_SOUDEX_MAP.get(ch, "") for ch in text if ch.isalpha()]
        first = next((c for c in text.upper() if c.isalpha()), "")
        out = [first]
        prev = _SOUDEX_MAP.get(first, "")
        for code in codes:
            if code and code != prev:
                out.append(code)
            if code:
                prev = code
            if len(out) == 4:
                break
        return ("".join(out) + "000")[:4]


# Ordered so the bitmask is stable and human-readable.
BLOCK_KEYS = ["phone10", "zip5", "zip3_name", "soundex", "name_prefix", "geo_cell", "token"]


@dataclass
class BlockConfig:
    """Tunables for candidate generation.

    max_block_size
        A key value shared by more than this many records is treated as too
        coarse and skipped. Without a cap, a key like ``zip3`` on a dense
        metro degrades into a near-full cross product, and one such block can
        dominate the whole candidate set. Skipped blocks are counted and
        reported so the cost is visible rather than silent.

    max_token_df_ratio
        A name token held by more than this fraction of Source 1 records is a
        category word ("auto", "shop"), not an identifier, so it is dropped from
        the token key. Keeps blocks small and recall on rare tokens high.
    """
    max_block_size: int = 40
    max_token_df_ratio: float = 0.01
    geo_cell_deg: float = 0.02  # ~2.2 km, enough to absorb geocoder jitter

    enabled: Dict[str, bool] = field(default_factory=lambda: {k: True for k in BLOCK_KEYS})


def _prefix_key(name_core: str, n: int = 4) -> str:
    return name_core[:n] if name_core else ""


def compute_block_keys(
    df: pd.DataFrame, cfg: BlockConfig, informative_tokens: Optional[Set[str]] = None
) -> pd.DataFrame:
    """Return a frame of blocking keys, one row per input row.

    Empty string means "this record cannot participate in that key", and is
    filtered out during generation rather than being treated as a shared value
    (otherwise every record with a missing phone would land in one block).
    """
    out = pd.DataFrame(index=df.index)
    name_core = df["name_core"].fillna("")
    zip5 = df["zip5"].fillna("")
    zip3 = df["zip3"].fillna("")

    out["phone10"] = df["phone10"].fillna("")
    out["zip5"] = zip5
    # zip3 anchored to the first 3 chars of the name core, so the block stays
    # small without needing the full zip.
    out["zip3_name"] = [
        f"{z}|{p}" if z and p else "" for z, p in zip(zip3, name_core.str[:3])
    ]
    out["soundex"] = [soundex(n) if n else "" for n in name_core]
    out["name_prefix"] = [_prefix_key(n) for n in name_core]

    # Geocell: round coordinates onto a grid. NaN -> "" so records without
    # coordinates simply do not produce geo candidates.
    cell = ""
    out["geo_cell"] = ""
    lat, lon = df["lat"], df["lon"]
    if "lat" in df and lat.notna().any():
        lat_i = (lat / cfg.geo_cell_deg).round()
        lon_i = (lon / cfg.geo_cell_deg).round()
        out["geo_cell"] = [
            f"{a}|{b}" if (pd.notna(a) and pd.notna(b)) else ""
            for a, b in zip(lat_i, lon_i)
        ]

    # Token key: the most valuable but noisiest. One row per (record, token)
    # is handled by the generator, which reads the column as a list.
    if informative_tokens:
        out["token"] = [
            tuple(sorted({t for t in toks if t in informative_tokens}))
            for toks in df["name_tokens"]
        ]
    else:
        out["token"] = [tuple(sorted(set(toks))) for toks in df["name_tokens"]]
    return out


def find_informative_tokens(s1: pd.DataFrame, cfg: BlockConfig) -> Set[str]:
    """Tokens rare enough to be identifiers rather than category words."""
    counts: Dict[str, int] = {}
    for toks in s1["name_tokens"]:
        for tok in set(toks):
            counts[tok] = counts.get(tok, 0) + 1
    if not s1.empty:
        cutoff = max(2, int(cfg.max_token_df_ratio * len(s1)))
        return {tok for tok, c in counts.items() if c <= cutoff}
    return set(counts)


def generate_candidates(
    s1_keys: pd.DataFrame,
    s1: pd.DataFrame,
    cand_keys_by_source: Dict[str, pd.DataFrame],
    cfg: BlockConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build the candidate pair set.

    Returns a frame with one row per (Source 1 record, candidate) pair, carrying
    the positional indices needed to join features, the originating source, and
    one boolean column per blocking key recording which keys produced the pair.
    """
    # (s1_pos, source, cand_pos) -> bitmask of contributing keys
    pair_mask: Dict[Tuple[int, str, int], int] = {}
    stats: Dict[str, object] = {
        "per_key": {k: {"pairs": 0, "blocks": 0, "skipped_blocks": 0} for k in BLOCK_KEYS},
    }

    for key in BLOCK_KEYS:
        if not cfg.enabled.get(key, True):
            continue
        bit = 1 << BLOCK_KEYS.index(key)
        s1_col = s1_keys[key]
        # Build candidate-side index: key value -> positions. `token` holds a
        # tuple per row so each element becomes its own index entry.
        cand_index: Dict[object, List[Tuple[str, int]]] = {}
        for source, ckeys in cand_keys_by_source.items():
            for pos, val in enumerate(ckeys[key]):
                if val is None or val == "":
                    continue
                values = val if key == "token" else (val,)
                for v in values:
                    if not v:
                        continue
                    cand_index.setdefault(v, []).append((source, pos))

        if key == "token":
            groups: Dict[object, List[int]] = {}
            for pos, val in enumerate(s1_col):
                if val is None:
                    continue
                for v in (val if isinstance(val, tuple) else (val,)):
                    if v:
                        groups.setdefault(v, []).append(pos)
        else:
            groups = {}
            for pos, val in enumerate(s1_col):
                if val is None or val == "":
                    continue
                groups.setdefault(val, []).append(pos)

        for value, s1_positions in groups.items():
            hits = cand_index.get(value)
            if not hits:
                continue
            stats["per_key"][key]["blocks"] += 1
            # A block this large is a sign the key is too coarse for this
            # slice of data, not that these records match. Skip and record it.
            if len(hits) > cfg.max_block_size:
                stats["per_key"][key]["skipped_blocks"] += 1
                continue
            n = 0
            for sp in s1_positions:
                for source, cp in hits:
                    pair_mask.setdefault((sp, source, cp), 0)
                    pair_mask[(sp, source, cp)] |= bit
                    n += 1
            stats["per_key"][key]["pairs"] += n

    if not pair_mask:
        return pd.DataFrame(
            columns=["s1_pos", "cand_pos", "candidate_source"] + BLOCK_KEYS
        ), stats

    keys = list(pair_mask.keys())
    out = pd.DataFrame(
        {
            "s1_pos": [k[0] for k in keys],
            "candidate_source": [k[1] for k in keys],
            "cand_pos": [k[2] for k in keys],
        }
    )
    masks = np.array([pair_mask[k] for k in keys], dtype=np.int64)
    for i, key in enumerate(BLOCK_KEYS):
        out[key] = (masks & (1 << i)) != 0
    out = out.sort_values(["s1_pos", "candidate_source", "cand_pos"]).reset_index(drop=True)
    stats["n_pairs"] = int(len(out))
    stats["n_pairs_per_s1"] = float(len(out) / max(1, len(s1)))
    return out, stats


def blocking_recall(
    candidates: pd.DataFrame,
    truth: Optional[pd.DataFrame],
    s1: pd.DataFrame,
    cand_frames: Dict[str, pd.DataFrame],
    s1_id_col: str,
    cand_id_cols: Dict[str, str],
) -> Dict[str, object]:
    """Fraction of known true pairs that survived blocking.

    This is the single most important diagnostic in the pipeline. A blocker with
    0.99 recall on 3% of the true pairs still drops hundreds of entities, and
    no downstream model can recover what blocking never generated.

    Reports:
      * overall recall
      * per-source recall
      * a per-key breakdown of *which* keys carried the surviving matches,
        which is what tells you which key you could afford to remove
      * the missed pairs themselves, so they can be inspected
    """
    if truth is None or truth.empty:
        return {"available": False, "reason": "no ground truth supplied"}

    # Position lookups must match how candidates were generated, i.e. positional
    # order within each cleaned frame.
    s1_pos_of = {rid: pos for pos, rid in enumerate(s1[s1_id_col])}
    cand_pos_of = {
        source: {rid: pos for pos, rid in enumerate(frame[cand_id_cols[source]])}
        for source, frame in cand_frames.items()
    }

    cand_key = {
        (int(r.s1_pos), r.candidate_source, int(r.cand_pos))
        for r in candidates.itertuples()
    }
    # Which keys fired for each surviving pair, for the per-key attribution.
    key_by_pair = {
        (int(r.s1_pos), r.candidate_source, int(r.cand_pos)): [
            k for k in BLOCK_KEYS if bool(getattr(r, k))
        ]
        for r in candidates.itertuples()
    }

    total = 0
    found = 0
    per_source: Dict[str, Dict[str, int]] = {}
    key_hits = {k: 0 for k in BLOCK_KEYS}
    missed = []

    for row in truth.itertuples():
        sp = s1_pos_of.get(row.source1_id)
        cp = cand_pos_of.get(row.candidate_source, {}).get(row.candidate_id)
        if sp is None or cp is None:
            # An id we cannot locate means the truth file and the data disagree.
            # Count it as a miss rather than silently skipping.
            total += 1
            continue
        total += 1
        bucket = per_source.setdefault(
            row.candidate_source, {"total": 0, "found": 0}
        )
        bucket["total"] += 1
        if (sp, row.candidate_source, cp) in cand_key:
            found += 1
            bucket["found"] += 1
            for k in key_by_pair.get((sp, row.candidate_source, cp), []):
                key_hits[k] += 1
        else:
            missed.append({
                "source1_id": row.source1_id,
                "candidate_id": row.candidate_id,
                "candidate_source": row.candidate_source,
            })

    return {
        "available": True,
        "n_truth": total,
        "n_survived": found,
        "recall": round(found / total, 4) if total else None,
        "per_source": {
            s: {
                "n_truth": b["total"],
                "n_survived": b["found"],
                "recall": round(b["found"] / b["total"], 4) if b["total"] else None,
            }
            for s, b in per_source.items()
        },
        # Attribution is not mutually exclusive: one surviving pair can be
        # reached by several keys.
        "keys_that_carried_matches": key_hits,
        "missed_pairs": pd.DataFrame(missed),
    }


# ===========================================================================
# Multi-index blocking for the real schema
# (entity_id / business_name / business_address / country)
#
# The implementation above answers the same question against a *rich* schema
# (phone, zip, geo). This section is the version for the schema that actually
# ships in `student_resource/dataset`: four columns, one of which is an opaque
# address blob, at 2.2M / 5.0M / 5.3M rows. Two structural differences:
#
#   * Multiple indexes instead of one. No single key over four fields has both
#     recall and a small block, so we build several and take their UNION (OR).
#     A record qualifies if ANY index fires -- AND would silently lose every
#     pair that only agrees on one of them.
#   * Discovery runs from the candidate side. Source 1 is indexed once and
#     Source 2/3 are read in chunks and probed against it, which is the only
#     orientation that keeps peak memory at "one Source 1 index + one chunk"
#     instead of "one joined table of 12.5M rows".
# ===========================================================================

# Rule identifiers. The order is also the tie-break order used when a row's
# merged candidate set has to be cut down to the cap: the earlier a rule sits,
# the more precise it is considered, so its candidates survive the cut.
RULE_COUNTRY_PREFIX = "country_prefix"
RULE_COUNTRY_CITY_PREFIX = "country_city_prefix"
RULE_SHARED_TOKEN = "shared_token"
RULE_PHONETIC_COUNTRY = "phonetic_country"
RULE_EXACT_ENTITY_ID = "exact_entity_id"

BLOCK_RULES: List[str] = [
    RULE_COUNTRY_PREFIX,
    RULE_COUNTRY_CITY_PREFIX,
    RULE_SHARED_TOKEN,
    RULE_PHONETIC_COUNTRY,
    RULE_EXACT_ENTITY_ID,
]

#: Default precision order used when the merged set exceeds the per-record cap.
DEFAULT_RULE_PRECEDENCE: List[str] = [
    RULE_COUNTRY_CITY_PREFIX,
    RULE_COUNTRY_PREFIX,
    RULE_PHONETIC_COUNTRY,
    RULE_SHARED_TOKEN,
    RULE_EXACT_ENTITY_ID,
]

# Rules contributed by the looser second pass (Change 4). They only ever ADD
# candidates; a pair that exists because of one of these still has to beat the
# F0.5-tuned threshold to become a match.
FALLBACK_FUZZY = "fallback_fuzzy_name"
FALLBACK_NEIGHBORHOOD = "fallback_sorted_neighborhood"

# The two fallback rules are appended to the bit space rather than reusing
# bits 0/1, so a fallback pair's `matched_rules` decodes unambiguously and the
# feature layer can tell "blocked by country+prefix" from "rescued by the
# fuzzy pass" without guessing from the label.
BLOCK_RULES += [FALLBACK_FUZZY, FALLBACK_NEIGHBORHOOD]

# A key value held by more than this many Source 1 rows is a category, not an
# identifier ("Paris" in France). Emitting it would cost a full cross product
# inside one bucket, so it is dropped and counted.
DEFAULT_MAX_BUCKET_SIZE = 400


def phonetic_code(text: str, scheme: str = "nysiis") -> str:
    """Phonetic code of a business name, '' when there is nothing to code.

    ``scheme`` is ``nysiis`` (default, more tolerant of transpositions than
    soundex) or ``soundex``. Falls back to soundex when jellyfish is
    unavailable, and to '' on a name with no ASCII letters -- a phonetic key
    built from an empty or non-Latin string is a key shared by every such
    record, which is the opposite of informative.
    """
    if not text:
        return ""
    try:
        from jellyfish import nysiis as _nysiis  # type: ignore
        from jellyfish import soundex as _soundex  # type: ignore

        code = _nysiis(text) if scheme == "nysiis" else _soundex(text)
        code = str(code or "").strip()
    except Exception:  # pragma: no cover - jellyfish missing
        code = soundex(text)
    # Non-Latin / vowel-only names come back empty or padded; treat both as
    # "no key" rather than as a shared value.
    return code if re.search(r"[A-Za-z]", text) and len(code) >= 3 else ""


@dataclass
class MultiIndexBlockConfig:
    """Tunables for the multi-index blocker. Every field is config-exposed.

    max_candidates_per_record
        Hard ceiling on merged candidates for one Source 2/3 record. Reached
        only after per-rule narrowing has already been tried, so a truncation
        here means the row genuinely matched a lot of coarse keys.

    per_rule_cap
        Soft ceiling applied *per rule* before merging. A rule above it is
        narrowed with an extra cheap key (see `generate_candidate_pairs`)
        rather than being dropped, because a dropped rule means a lost record.

    max_bucket_size
        A key value carried by more than this many Source 1 rows is
        uninformative and skipped for that value.

    max_token_df_ratio / max_token_df_absolute
        A name token held by more than (ratio x |Source 1|) or more than
        `absolute` Source 1 rows is a category word ("services", "ltd") and
        never becomes a blocking key. Both bounds are needed: the absolute one
        keeps a small sample from calling every token informative.

    scope_token_index_by_country
        True (default) keys the token index as ``country|token``, so a "Paris"
        in France and a "Paris" in Texas cannot meet. Set False to use the bare
        global token index; that raises recall for records whose country label
        is wrong at the cost of large cross-country blocks.

    fallback_*
        Second pass for rows that got zero candidates. Its thresholds are for
        *candidate generation only* and never feed a match decision.
    """

    # --- per-record safety (Change 3) ---------------------------------------
    max_candidates_per_record: int = 100
    per_rule_cap: int = 40
    large_candidate_multiplier: float = 2.0  # > cap x this => debug-logged

    # --- index construction -------------------------------------------------
    name_prefix_len: int = 4
    phonetic_scheme: str = "nysiis"
    min_token_len: int = 3
    max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE
    max_token_df_ratio: float = 0.002
    max_token_df_absolute: int = 2000
    scope_token_index_by_country: bool = True
    rule_precedence: List[str] = field(default_factory=lambda: list(DEFAULT_RULE_PRECEDENCE))

    # --- rule switches ------------------------------------------------------
    enable_country_prefix: bool = True
    enable_country_city_prefix: bool = True
    enable_shared_token: bool = True
    enable_phonetic_country: bool = True
    #: Exact entity_id index. Kept as a configurable hook and OFF by default:
    #: Source 1 and Source 2/3 ids come from independent registries, so it is
    #: not expected to ever fire -- but it costs one dict and would settle the
    #: question empirically on a dataset that does share identifiers.
    enable_exact_entity_id: bool = False

    # --- fallback pass (Change 4) -------------------------------------------
    enable_fallback: bool = True
    fallback_fuzzy_floor: float = 0.72
    fallback_window: int = 15
    fallback_neighborhood_window: int = 3
    fallback_max_candidates: int = 50
    fallback_min_token_overlap: int = 1

    # --- logging ------------------------------------------------------------
    enable_debug_log: bool = True
    max_debug_rows: int = 200_000


@dataclass
class Source1Indexes:
    """The Source 1 side of the blocker: several inverted indexes, one per rule.

    Every index is ``key -> list[entity_id]`` built with ``defaultdict(list)``.
    Keys are only ever created from a *non-empty* value, so a record with a
    blank country, blank name or unparseable address simply produces no entry
    instead of joining a shared "" bucket.
    """

    country_prefix: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    country_city_prefix: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    token: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    phonetic_country: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    entity_id: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    #: country -> sorted normalized names, and the ids aligned to them. This is
    #: the structure the sorted-neighborhood fallback pass walks.
    sorted_names: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    sorted_ids: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    #: entity_id -> positional row in the Source 1 frame, so pair features can
    #: be gathered without a per-pair .loc lookup.
    s1_row_of_id: Dict[str, int] = field(default_factory=dict)
    token_df: Dict[str, int] = field(default_factory=dict)
    config: MultiIndexBlockConfig = field(default_factory=MultiIndexBlockConfig)
    stats: Dict[str, object] = field(default_factory=dict)

    def rule_hit_counts(self) -> Dict[str, int]:
        """How many entity_ids each index bucket would emit in total."""
        return {
            RULE_COUNTRY_PREFIX: sum(len(v) for v in self.country_prefix.values()),
            RULE_COUNTRY_CITY_PREFIX: sum(len(v) for v in self.country_city_prefix.values()),
            RULE_SHARED_TOKEN: sum(len(v) for v in self.token.values()),
            RULE_PHONETIC_COUNTRY: sum(len(v) for v in self.phonetic_country.values()),
            RULE_EXACT_ENTITY_ID: sum(len(v) for v in self.entity_id.values()),
        }


@dataclass
class CandidateBatch:
    """One chunk's worth of candidate generation.

    pairs
        One row per (Source 1 entity, Source 2/3 record) pair that qualified.
        Carries the positional indices used by `compute_pairwise_features` and
        the per-pair rule bitmask.
    records
        One row per Source 2/3 record in the chunk -- including the ones with
        zero candidates, which is what makes "no record is silently dropped"
        checkable rather than aspirational.
    debug
        Rows for the debug log: oversized candidate sets, truncations,
        zero-candidate rows and records with no usable blocking key at all.
    """

    pairs: pd.DataFrame
    records: pd.DataFrame
    debug: pd.DataFrame
    stats: Dict[str, object] = field(default_factory=dict)


def _column(frame: pd.DataFrame, name: str, default=object()) -> np.ndarray:
    """Column as a numpy array, or a uniform default when absent.

    Candidates are probed with a plain array of strings rather than pandas
    indexing: the inner loop touches every column of every row, and per-row
    attribute lookup dominates the cost otherwise.
    """
    if name in frame.columns:
        return frame[name].to_numpy()
    return np.full(len(frame), default, dtype=object)


def build_source1_indexes(
    s1: pd.DataFrame,
    cfg: Optional[MultiIndexBlockConfig] = None,
    id_col: str = "entity_id",
) -> Source1Indexes:
    """Build every Source 1 index from the four real columns.

    Reads `name_core` / `name_tokens` / `city_norm` / `country_norm` as written
    by `normalize.clean_entity_frame`; falls back to normalizing `business_name`
    / `business_address` on the fly when those derived columns are absent, so
    the function is usable on a raw frame in tests and one-off scripts.

    A row contributes to a rule only when every part of that rule's key is
    present. This is the single most important line of the whole blocker: an
    empty key is a key shared by every record that lacks that field, so using
    one would turn a missing value into a mega-block.
    """
    cfg = cfg or MultiIndexBlockConfig()

    # Accept either a cleaned frame or a raw one.
    if "name_core" not in s1.columns:
        cleaned = _clean_minimal(s1)
    else:
        cleaned = s1

    names = _column(cleaned, "name_core", "")
    countries = _column(cleaned, "country_norm", "")
    cities = _column(cleaned, "city_norm", "")
    ids = _column(cleaned, id_col, None)
    # Hoisted out of the loop below: `_column` calls `to_numpy()`, which copies
    # the whole array, so calling it per row would be quadratic.
    token_lists = _column(cleaned, "name_tokens", ())

    # ---- pass 1: document frequency, so "informative" is judged by Source 1 --
    df: Dict[str, int] = defaultdict(int)
    for toks in token_lists:
        for tok in set(toks or ()):
            df[tok] += 1
    n_s1 = len(cleaned)
    token_cutoff = min(
        max(2, int(cfg.max_token_df_ratio * max(1, n_s1))), cfg.max_token_df_absolute
    )

    idx = Source1Indexes(config=cfg, token_df=dict(df))
    prefix_len = max(1, int(cfg.name_prefix_len))
    scope_token = bool(cfg.scope_token_index_by_country)

    names_by_country: Dict[str, List[str]] = defaultdict(list)
    ids_by_country: Dict[str, List[str]] = defaultdict(list)

    stats = {
        "n_source1": int(n_s1),
        "token_df_cutoff": int(token_cutoff),
        "n_tokens_distinct": len(df),
        "n_tokens_informative": 0,
        "rows_without_country": 0,
        "rows_without_name": 0,
        "rows_without_city": 0,
        "rows_unreachable": 0,  # no index at all -> can never be proposed
    }

    for row in range(n_s1):
        name = names[row] or ""
        country = countries[row] or ""
        city = cities[row] or ""
        eid = ids[row]
        if eid is None or (isinstance(eid, float) and eid != eid):
            continue

        if not country:
            stats["rows_without_country"] += 1
        if not name:
            stats["rows_without_name"] += 1
        if not city:
            stats["rows_without_city"] += 1

        placed = False
        if country and name:
            if cfg.enable_country_prefix:
                idx.country_prefix[f"{country}|{name[:prefix_len]}"].append(eid)
                placed = True
            if cfg.enable_phonetic_country:
                code = phonetic_code(name, cfg.phonetic_scheme)
                if code:
                    idx.phonetic_country[f"{country}|{code}"].append(eid)
                    placed = True
            if cfg.enable_country_city_prefix and city:
                idx.country_city_prefix[f"{country}|{city}|{name[:prefix_len]}"].append(eid)
                placed = True
            names_by_country[country].append(name)
            ids_by_country[country].append(eid)

        if cfg.enable_shared_token:
            for tok in set(token_lists[row] or ()):
                # Only tokens that are rare enough to identify a business become
                # keys; everything else is a category word.
                if len(tok) < cfg.min_token_len or df.get(tok, 0) > token_cutoff:
                    continue
                key = f"{country}|{tok}" if (scope_token and country) else tok
                idx.token[key].append(eid)
                placed = True

        if cfg.enable_exact_entity_id:
            idx.entity_id[str(eid)].append(eid)
            placed = True

        if not placed:
            stats["rows_unreachable"] += 1

    stats["n_tokens_informative"] = len(idx.token)
    stats["bucket_counts"] = {
        "country_prefix": len(idx.country_prefix),
        "country_city_prefix": len(idx.country_city_prefix),
        "token": len(idx.token),
        "phonetic_country": len(idx.phonetic_country),
        "entity_id": len(idx.entity_id),
    }
    stats["rule_hit_counts"] = idx.rule_hit_counts()

    # Bucket cap: a key value held by too many Source 1 rows is noise. Done
    # here rather than at query time so a hot bucket is never even materialised
    # for a query that has to test it anyway.
    for attr, name in (
        ("country_prefix", RULE_COUNTRY_PREFIX),
        ("country_city_prefix", RULE_COUNTRY_CITY_PREFIX),
        ("phonetic_country", RULE_PHONETIC_COUNTRY),
    ):
        store = getattr(idx, attr)
        dropped = sum(1 for v in store.values() if len(v) > cfg.max_bucket_size)
        for key in [k for k, v in store.items() if len(v) > cfg.max_bucket_size]:
            del store[key]
        stats[f"dropped_hot_buckets_{name}"] = dropped
    dropped_tokens = sum(1 for v in idx.token.values() if len(v) > cfg.max_bucket_size)
    for key in [k for k, v in idx.token.items() if len(v) > cfg.max_bucket_size]:
        del idx.token[key]
    stats["dropped_hot_buckets_shared_token"] = dropped_tokens

    # ---- sorted-neighborhood structure for the fallback pass ----------------
    for country, values in names_by_country.items():
        pairs = sorted(zip(values, ids_by_country[country]))
        idx.sorted_names[country] = [p[0] for p in pairs]
        idx.sorted_ids[country] = [p[1] for p in pairs]

    for row, eid in enumerate(ids):
        if eid is not None:
            idx.s1_row_of_id[eid] = row

    idx.stats = stats
    return idx


def _clean_minimal(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw real-schema frame on the fly (used by tests/one-offs)."""
    from .normalize import clean_entity_frame

    return clean_entity_frame(
        frame,
        {"name": "business_name", "address": "business_address", "country": "country"},
    )


def _rule_bit(rule: str) -> int:
    return 1 << BLOCK_RULES.index(rule)


def _rules_from_mask(mask: int) -> str:
    return "|".join(r for i, r in enumerate(BLOCK_RULES) if mask & (1 << i))


def _narrow_by_shared_token(
    candidates: List[str],
    query_tokens: set,
    row_of_id: Dict[str, int],
    s1_token_lists: np.ndarray,
    min_overlap: int = 1,
) -> List[str]:
    """Keep only candidates sharing at least ``min_overlap`` name tokens.

    This is the extra cheap key used to tame an over-producing rule: requiring
    one shared token turns "every 'Acme' in the country" into "the 'Acme's
    whose name also contains a word of mine", which is a far smaller set at
    essentially no cost (one set intersection per candidate).
    """
    if not candidates or not query_tokens:
        return list(candidates)
    keep = []
    for eid in candidates:
        row = row_of_id.get(eid)
        if row is None:
            continue
        other = set(s1_token_lists[row] or ())
        if other and len(other & query_tokens) >= min_overlap:
            keep.append(eid)
    return keep


def generate_candidate_pairs(
    chunk: pd.DataFrame,
    indexes: Source1Indexes,
    id_col: str = "entity_id",
    source_dataset: str = "source2",
    cfg: Optional[MultiIndexBlockConfig] = None,
) -> CandidateBatch:
    """Probe one Source 2/3 chunk against every Source 1 index (OR logic).

    A Source 2/3 record becomes a candidate pair with a Source 1 entity if ANY
    enabled rule fires -- country+name-prefix, country+city+name-prefix, a
    shared significant name token, or phonetic+country. Rules are OR'd, not
    AND'd: an AND would keep only pairs that agree on every key, which is
    precisely the set that needs no ensemble in the first place.

    Returns a `CandidateBatch`:
      * ``pairs``   one row per qualifying (Source 1, Source 2/3) pair, with the
                    positional indices and the rule bitmask.
      * ``records`` one row per input record, including the zero-candidate
                    ones, so downstream stages can see what was unresolved.
      * ``debug``   oversized / truncated / zero-candidate diagnostics.
    """
    cfg = cfg or indexes.config or MultiIndexBlockConfig()
    n = len(chunk)
    if n == 0:
        return _empty_batch()

    names = _column(chunk, "name_core", "")
    countries = _column(chunk, "country_norm", "")
    cities = _column(chunk, "city_norm", "")
    ids = _column(chunk, id_col, None)
    token_lists = _column(chunk, "name_tokens", ())
    # Source 1 token lists are needed to narrow an over-producing rule. They are
    # attached once by `attach_source1_frame`; when absent (raw-frame call sites)
    # narrowing degrades to "no extra key available" instead of failing.
    s1_token_lists = indexes.stats.get("s1_name_tokens", np.empty(0, dtype=object))
    # entity_id -> chunk position, so the per-pair row index is a dict lookup.
    pos_of = {str(v): i for i, v in enumerate(ids) if v is not None}

    prefix_len = max(1, int(cfg.name_prefix_len))
    scope_token = bool(cfg.scope_token_index_by_country)
    precedence = {r: i for i, r in enumerate(cfg.rule_precedence)}
    rule_bits = {r: _rule_bit(r) for r in BLOCK_RULES}
    enabled = {
        RULE_COUNTRY_PREFIX: cfg.enable_country_prefix,
        RULE_COUNTRY_CITY_PREFIX: cfg.enable_country_city_prefix,
        RULE_SHARED_TOKEN: cfg.enable_shared_token,
        RULE_PHONETIC_COUNTRY: cfg.enable_phonetic_country,
        RULE_EXACT_ENTITY_ID: cfg.enable_exact_entity_id,
    }
    large_threshold = max(cfg.max_candidates_per_record + 1,
                          int(cfg.max_candidates_per_record * cfg.large_candidate_multiplier))

    pairs_s1: List[str] = []
    pairs_cand: List[str] = []
    pairs_mask: List[int] = []
    rec_ids: List[str] = []
    rec_counts: List[int] = []
    rec_reasons: List[str] = []
    rec_status: List[str] = []
    rec_notes: List[str] = []
    debug_rows: List[Dict[str, object]] = []
    rule_pair_counts: Dict[str, int] = {r: 0 for r in BLOCK_RULES}
    n_zero = 0
    n_truncated = 0
    n_narrowed = 0
    n_large = 0

    for row in range(n):
        rid = ids[row]
        if rid is None or (isinstance(rid, float) and rid != rid):
            # A record with no id cannot be reported, let alone matched.
            continue
        rid = str(rid)
        name = names[row] or ""
        country = countries[row] or ""
        city = cities[row] or ""
        raw_tokens = [t for t in (token_lists[row] or ()) if len(t) >= cfg.min_token_len]
        query_tokens = set(raw_tokens)

        # entity_id -> bitmask of the rules that produced it
        found: Dict[str, int] = {}
        per_rule: Dict[str, int] = {}

        def add(ids_list, rule: str) -> None:
            if not ids_list:
                return
            bit = rule_bits[rule]
            for eid in ids_list:
                prev = found.get(eid, 0)
                found[eid] = prev | bit
                per_rule[rule] = per_rule.get(rule, 0) + 1

        # -- rule 1: country + name prefix ---------------------------------
        if enabled[RULE_COUNTRY_PREFIX] and country and name:
            add(indexes.country_prefix.get(f"{country}|{name[:prefix_len]}"),
                RULE_COUNTRY_PREFIX)
        # -- rule 2: country + extracted city + name prefix ------------------
        if enabled[RULE_COUNTRY_CITY_PREFIX] and country and name and city:
            add(indexes.country_city_prefix.get(f"{country}|{city}|{name[:prefix_len]}"),
                RULE_COUNTRY_CITY_PREFIX)
        # -- rule 3: shared significant name token --------------------------
        if enabled[RULE_SHARED_TOKEN] and query_tokens:
            for tok in query_tokens:
                key = f"{country}|{tok}" if (scope_token and country) else tok
                hits = indexes.token.get(key)
                if hits and len(hits) <= cfg.max_bucket_size:
                    add(hits, RULE_SHARED_TOKEN)
                elif hits:
                    # Hot bucket: too coarse to be evidence. Narrow it with a
                    # cheap second key instead of emitting thousands of pairs.
                    narrowed = _narrow_by_shared_token(
                        hits, query_tokens, indexes.s1_row_of_id, s1_token_lists
                    )
                    if len(narrowed) > cfg.max_bucket_size:
                        narrowed = narrowed[: cfg.max_bucket_size]
                    add(narrowed, RULE_SHARED_TOKEN)
        # -- rule 4: phonetic code + country --------------------------------
        if enabled[RULE_PHONETIC_COUNTRY] and country and name:
            code = phonetic_code(name, cfg.phonetic_scheme)
            if code:
                add(indexes.phonetic_country.get(f"{country}|{code}"),
                    RULE_PHONETIC_COUNTRY)
        # -- rule 5: exact entity_id hook (off by default) -------------------
        if enabled[RULE_EXACT_ENTITY_ID]:
            add(indexes.entity_id.get(rid), RULE_EXACT_ENTITY_ID)

        notes: List[str] = []
        total = len(found)

        # ---- Change 3: candidate-size safety ------------------------------
        # An over-producing rule is narrowed, never dropped: dropping it loses
        # a real record, narrowing it loses only the unidentifiable tail.
        if total > cfg.max_candidates_per_record:
            for rule in cfg.rule_precedence:
                if per_rule.get(rule, 0) <= cfg.per_rule_cap:
                    continue
                bit = rule_bits[rule]
                rule_ids = [eid for eid, m in found.items() if m & bit]
                narrowed = _narrow_by_shared_token(
                    rule_ids, query_tokens, indexes.s1_row_of_id, s1_token_lists
                )
                if not narrowed:
                    continue
                if len(narrowed) < len(rule_ids):
                    n_narrowed += 1
                    notes.append(f"narrowed:{rule}:{len(rule_ids)}->{len(narrowed)}")
                # Drop the rule's contribution, then re-add only the survivors.
                for eid in rule_ids:
                    m = found[eid] & ~bit
                    if m:
                        found[eid] = m
                    else:
                        del found[eid]
                add(narrowed, rule)
                if len(found) <= cfg.max_candidates_per_record:
                    break

        total = len(found)
        # Still over the hard cap: cut by *evidence strength*, deterministically.
        # Truncation is logged, never silent.
        #
        # This used to sort by best-rule precedence, which is a systematic error
        # rather than a random tail: every candidate that some broad rule
        # happened to touch sorted ahead of a candidate whose only evidence was
        # a shared rare token, so true matches were preferentially discarded.
        # Measured effect at name_prefix_len=4: blocking recall 0.035.
        #
        # Ranking is now (1) most independent rules agreeing, then (2) most
        # specific single rule, then (3) entity_id for determinism. A candidate
        # corroborated by two independent keys now outranks one matched by a
        # single broad key regardless of which rules those were.
        truncated = False
        if total > cfg.max_candidates_per_record:
            def _rank(kv):
                mask = kv[1]
                n_rules = sum(1 for r, b in rule_bits.items() if mask & b)
                best_spec = min(
                    (precedence.get(r, 99) for r, b in rule_bits.items() if mask & b),
                    default=99,
                )
                return (-n_rules, best_spec, kv[0])

            ordered = sorted(found.items(), key=_rank)
            notes.append(f"truncated:{total}->{cfg.max_candidates_per_record}")
            found = dict(ordered[: cfg.max_candidates_per_record])
            total = len(found)
            n_truncated += 1
            truncated = True
        if total > cfg.max_candidates_per_record:  # pragma: no cover - defensive
            total = cfg.max_candidates_per_record

        # ---- emit ---------------------------------------------------------
        reasons: set = set()
        for eid, mask in found.items():
            pairs_s1.append(eid)
            pairs_cand.append(rid)
            pairs_mask.append(mask)
            for r in BLOCK_RULES:
                if mask & rule_bits[r]:
                    reasons.add(r)
        for rule, count in per_rule.items():
            rule_pair_counts[rule] += count

        if total:
            rec_status.append("resolved")
        else:
            rec_status.append("unresolved")
            n_zero += 1
            reasons = set()
        if total > large_threshold:
            n_large += 1
            notes.append("large_candidate_set")
            if cfg.enable_debug_log:
                debug_rows.append({
                    "source_record_id": rid, "source_dataset": source_dataset,
                    "event": "large_candidate_set", "candidate_count": total,
                    "cap": cfg.max_candidates_per_record,
                    "per_rule_hits": "|".join(f"{k}={v}" for k, v in sorted(per_rule.items())),
                    "note": ";".join(notes) if notes else "",
                })
        elif truncated and cfg.enable_debug_log:
            # A truncation is a decision the reviewer has to be able to see, so
            # it gets its own row: this is the record where candidates were
            # dropped to fit the cap, and "which ones" is not answerable from
            # the output files.
            debug_rows.append({
                "source_record_id": rid, "source_dataset": source_dataset,
                "event": "truncated_to_cap", "candidate_count": total,
                "cap": cfg.max_candidates_per_record,
                "per_rule_hits": "|".join(f"{k}={v}" for k, v in sorted(per_rule.items())),
                "note": ";".join(notes) if notes else "",
            })

        rec_ids.append(rid)
        rec_counts.append(total)
        rec_reasons.append("|".join(sorted(reasons)) if reasons else "none")
        rec_notes.append(";".join(notes))

    pairs = pd.DataFrame({
        "s1_entity_id": pairs_s1,
        "source_record_id": pairs_cand,
        "source_dataset": source_dataset,
        "rule_bitmask": np.asarray(pairs_mask, dtype=np.int64),
        "matched_rules": [_rules_from_mask(m) for m in pairs_mask],
        "s1_row": np.asarray(
            [indexes.s1_row_of_id.get(e, -1) for e in pairs_s1], dtype=np.int64),
        "cand_pos": np.asarray([pos_of.get(r, -1) for r in pairs_cand], dtype=np.int64),
    })
    records = pd.DataFrame({
        "source_record_id": rec_ids,
        "source_dataset": source_dataset,
        "candidate_count": np.asarray(rec_counts, dtype=np.int64),
        "candidate_generation_reasons": rec_reasons,
        "generation_status": rec_status,
        "processing_notes": rec_notes,
    })
    stats = {
        "n_records": int(n),
        "n_records_resolved": int(len(rec_ids) - n_zero),
        "n_records_unresolved": int(n_zero),
        "n_pairs": int(len(pairs_s1)),
        "mean_candidates_per_record": float(len(pairs_s1) / max(1, len(rec_ids))),
        "max_candidate_count": int(max(rec_counts) if rec_counts else 0),
        "n_rows_over_cap_narrowed": int(n_narrowed),
        "n_rows_truncated": int(n_truncated),
        "n_rows_large_candidate_set": int(n_large),
        "pairs_per_rule": rule_pair_counts,
    }
    debug = _debug_frame(debug_rows, cfg)
    return CandidateBatch(pairs=pairs, records=records, debug=debug, stats=stats)


_EMPTY = pd.DataFrame()


def _empty_batch() -> CandidateBatch:
    empty_pairs = pd.DataFrame({
        "s1_entity_id": [], "source_record_id": [], "source_dataset": [],
        "rule_bitmask": [], "matched_rules": [], "s1_row": [], "cand_pos": [],
    })
    empty_records = pd.DataFrame({
        "source_record_id": [], "source_dataset": [], "candidate_count": [],
        "candidate_generation_reasons": [], "generation_status": [],
        "processing_notes": [],
    })
    return CandidateBatch(pairs=empty_pairs, records=empty_records,
                          debug=pd.DataFrame(), stats={})


def _debug_frame(rows: List[Dict[str, object]], cfg: MultiIndexBlockConfig) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "source_record_id", "source_dataset", "event", "candidate_count",
            "cap", "per_rule_hits", "note",
        ])
    frame = pd.DataFrame(rows)
    if cfg.max_debug_rows and len(frame) > cfg.max_debug_rows:
        frame = frame.head(cfg.max_debug_rows)
    return frame


def fallback_candidate_generation(
    unresolved: pd.DataFrame,
    chunk: pd.DataFrame,
    indexes: Source1Indexes,
    id_col: str = "entity_id",
    source_dataset: str = "source2",
    cfg: Optional[MultiIndexBlockConfig] = None,
) -> CandidateBatch:
    """Second, looser pass for records that got zero candidates (Change 4).

    Only ever applied to rows the first pass left unresolved, and only ever
    *adds* candidates. Its thresholds are candidate-generation thresholds: a
    pair that exists because of this pass still has to beat the same
    F0.5-tuned classifier threshold as any other pair, so loosening here cannot
    manufacture a match.

    Two mechanisms, both country-scoped and both O(log n) per record:

    ``fallback_fuzzy_name``
        Sorted-neighborhood scan: binary-search the record's normalized name
        into its country's sorted name list, take +/- ``fallback_window``
        neighbours, and keep those whose ``token_set_ratio`` clears
        ``fallback_fuzzy_floor`` -- a bar low enough to catch real matches the
        blocking keys missed, and never used for a match decision.

    ``fallback_sorted_neighborhood``
        The immediate +/- ``fallback_neighborhood_window`` neighbours with no
        similarity floor at all. Catches the record whose name was mangled past
        the fuzzy bar but still sorts next to its twin.

    Returns a `CandidateBatch` whose ``records`` frame marks each still-empty
    row ``unresolved_after_fallback`` so the caller can route it to manual
    review instead of dropping it.
    """
    cfg = cfg or indexes.config or MultiIndexBlockConfig()
    if unresolved is None or len(unresolved) == 0:
        return _empty_batch()

    pos_of = {str(v): i for i, v in enumerate(_column(chunk, id_col, None))}
    names = _column(chunk, "name_core", "")
    countries = _column(chunk, "country_norm", "")

    pairs_s1: List[str] = []
    pairs_cand: List[str] = []
    pairs_mask: List[int] = []
    rec_ids: List[str] = []
    rec_counts: List[int] = []
    rec_reasons: List[str] = []
    rec_status: List[str] = []
    rec_notes: List[str] = []
    debug_rows: List[Dict[str, object]] = []
    n_rescued = 0

    # Rule bits come from BLOCK_RULES, so a fallback pair's mask decodes to the
    # fallback labels in `matched_rules` and never collides with a real rule.
    fb_bit = _rule_bit(FALLBACK_FUZZY)
    nb_bit = _rule_bit(FALLBACK_NEIGHBORHOOD)
    for rid in unresolved["source_record_id"].astype(str):
        pos = pos_of.get(rid)
        if pos is None:
            continue
        name = names[pos] or ""
        country = countries[pos] or ""
        if not name or not country:
            rec_status.append("unresolved_after_fallback")
            rec_ids.append(rid)
            rec_counts.append(0)
            rec_reasons.append("none")
            rec_notes.append("no_name_or_country")
            continue

        sorted_names = indexes.sorted_names.get(country)
        if not sorted_names:
            rec_status.append("unresolved_after_fallback")
            rec_ids.append(rid)
            rec_counts.append(0)
            rec_reasons.append("none")
            rec_notes.append("no_source1_rows_in_country")
            continue

        lo = bisect_left(sorted_names, name)
        hits: Dict[str, int] = {}
        n_tok = max(1, cfg.fallback_min_token_overlap)

        # -- path A: sorted neighborhood + fuzzy floor -----------------------
        start = max(0, lo - cfg.fallback_window)
        stop = min(len(sorted_names), lo + cfg.fallback_window + 1)
        neighbor_ids = indexes.sorted_ids[country]
        for j in range(start, stop):
            score = fuzz.token_set_ratio(name, sorted_names[j])
            if score >= 100.0 * cfg.fallback_fuzzy_floor:
                hits[neighbor_ids[j]] = hits.get(neighbor_ids[j], 0) | fb_bit

        # -- path B: immediate neighbors, no similarity floor ---------------
        for j in range(max(0, lo - cfg.fallback_neighborhood_window),
                       min(len(sorted_names), lo + cfg.fallback_neighborhood_window + 1)):
            hits[neighbor_ids[j]] = hits.get(neighbor_ids[j], 0) | nb_bit

        notes: List[str] = []
        if len(hits) > cfg.fallback_max_candidates:
            keys = sorted(hits)[: cfg.fallback_max_candidates]
            notes.append(f"fallback_truncated:{len(hits)}->{len(keys)}")
            hits = {k: hits[k] for k in keys}

        for eid, mask in hits.items():
            pairs_s1.append(eid)
            pairs_cand.append(rid)
            pairs_mask.append(mask)

        reasons = []
        if hits:
            n_rescued += 1
            rec_status.append("resolved_by_fallback")
            if any(m & fb_bit for m in hits.values()):
                reasons.append(FALLBACK_FUZZY)
            if any(m & nb_bit for m in hits.values()):
                reasons.append(FALLBACK_NEIGHBORHOOD)
        else:
            rec_status.append("unresolved_after_fallback")
            reasons = ["none"]
            if cfg.enable_debug_log:
                debug_rows.append({
                    "source_record_id": rid, "source_dataset": source_dataset,
                    "event": "unresolved_after_fallback", "candidate_count": 0,
                    "cap": cfg.fallback_max_candidates, "per_rule_hits": "",
                    "note": "no_fuzzy_or_neighbor_match",
                })
        rec_ids.append(rid)
        rec_counts.append(len(hits))
        rec_reasons.append("|".join(reasons))
        rec_notes.append(";".join(notes))

    pairs = pd.DataFrame({
        "s1_entity_id": pairs_s1,
        "source_record_id": pairs_cand,
        "source_dataset": source_dataset,
        "rule_bitmask": np.asarray(pairs_mask, dtype=np.int64),
        "matched_rules": [_rules_from_mask(m) for m in pairs_mask],
        "s1_row": np.asarray(
            [indexes.s1_row_of_id.get(e, -1) for e in pairs_s1], dtype=np.int64),
        "cand_pos": np.asarray([pos_of.get(r, -1) for r in pairs_cand], dtype=np.int64),
    })
    records = pd.DataFrame({
        "source_record_id": rec_ids,
        "source_dataset": source_dataset,
        "candidate_count": np.asarray(rec_counts, dtype=np.int64),
        "candidate_generation_reasons": rec_reasons,
        "generation_status": rec_status,
        "processing_notes": rec_notes,
    })
    debug = _debug_frame(debug_rows, cfg)
    stats = {
        "n_input_unresolved": int(len(unresolved)),
        "n_rescued": int(n_rescued),
        "n_still_unresolved": int(len(rec_status) - n_rescued),
        "n_pairs": int(len(pairs_s1)),
    }
    return CandidateBatch(pairs=pairs, records=records, debug=debug, stats=stats)


def attach_source1_frame(indexes: Source1Indexes, s1: pd.DataFrame) -> Source1Indexes:
    """Cache the Source 1 token column on the indexes for rule narrowing.

    Called once after `build_source1_indexes`; keeps the per-pair narrowing
    lookup a numpy take instead of a DataFrame lookup, without duplicating the
    whole frame inside the index object.
    """
    indexes.stats["s1_name_tokens"] = _column(s1, "name_tokens", ())
    return indexes

