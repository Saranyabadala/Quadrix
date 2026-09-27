"""End-to-end orchestration.

Stages, each independently runnable and each writing its artifacts:

  1. generate  synthetic source1/2/3 + hidden truth   (optional, benchmark only)
  2. clean     normalize all three sources
  3. block     candidate generation + blocking recall
  4. features  pairwise feature matrix
  5. labels    silver labels + stratified manual-labeling sheet
  6. train     fit the match classifier, tune the threshold
  7. assemble  score everything, group per entity, write the final table

The pipeline is linear on purpose. Each stage reads what the previous one
wrote, so a stage can be re-run in isolation while iterating -- which is how you
actually tune a blocking key or a feature set without re-deriving everything
downstream.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import (assemble, block, data, evaluate, features as feat, generate,
               hardneg, labels, model)
from .block import BlockConfig
from .features import feature_columns


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _write_json(path: str, payload: dict) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=default)


# ---------------------------------------------------------------------------
# Stage 1: generate
# ---------------------------------------------------------------------------

def stage_generate(n_entities: int, raw_dir: str, truth_dir: str, seed: int) -> Dict:
    _ensure_dirs(raw_dir, truth_dir)
    frames = generate.generate(
        n_entities=n_entities, out_dir=raw_dir, truth_dir=truth_dir, seed=seed
    )
    print(f"  wrote {len(frames['source1'])} source1 / {len(frames['source2'])} source2 "
          f"/ {len(frames['source3'])} source3 rows")
    return {"n_entities": n_entities, **{k: len(v) for k, v in frames.items()}}


# ---------------------------------------------------------------------------
# Stages 2-4: clean, block, features
# ---------------------------------------------------------------------------

def stage_prepare(
    config_path: str,
    root: str,
    work_dir: str,
    block_cfg: BlockConfig,
    truth_path: Optional[str] = None,
) -> Dict:
    _ensure_dirs(work_dir)

    config = data.load_config(config_path)
    print("[2/7] cleaning and normalizing")
    cleaned = data.load_sources(config, root=root)

    s1 = cleaned["source1"]
    s1_id_col = config["sources"]["source1"]["id_col"]
    cand_sources = [s for s in cleaned if s != "source1"]
    cand_by_source = {s: cleaned[s] for s in cand_sources}
    cand_id_cols = {s: config["sources"][s]["id_col"] for s in cand_sources}

    print("[3/7] blocking / candidate generation")
    informative = block.find_informative_tokens(s1, block_cfg)
    s1_keys = block.compute_block_keys(s1, block_cfg, informative)
    cand_keys = {
        s: block.compute_block_keys(cand_by_source[s], block_cfg, informative)
        for s in cand_sources
    }
    candidates, block_stats = block.generate_candidates(s1_keys, s1, cand_keys, block_cfg)

    n_full = len(s1) * sum(len(cand_by_source[s]) for s in cand_sources)
    print(f"  {n_full:,} full cross product -> {len(candidates):,} candidate pairs "
          f"({len(candidates) / max(1, n_full):.4%} of the space, "
          f"{len(candidates) / max(1, len(s1)):.1f} per Source 1 entity)")

    recall = None
    truth = data.load_truth(truth_path) if truth_path else None
    if truth is not None:
        recall = block.blocking_recall(
            candidates, truth, s1, cand_by_source, s1_id_col, cand_id_cols
        )
        if recall.get("available"):
            print(f"  BLOCKING RECALL: {recall['recall']:.4f} "
                  f"({recall['n_survived']}/{recall['n_truth']} true pairs survived)")
            for src, r in recall["per_source"].items():
                print(f"    {src}: {r['recall']:.4f} "
                      f"({r['n_survived']}/{r['n_truth']})")
            print(f"    keys that carried matches: {recall['keys_that_carried_matches']}")
        else:
            print("  blocking recall: unavailable (no ground truth supplied)")

    print("[4/7] feature engineering")
    fdf = feat.compute_features(s1, cand_by_source, candidates, s1_id_col, cand_id_cols)
    fcols = feature_columns(fdf)
    print(f"  {len(fcols)} features over {len(fdf):,} candidate pairs")

    # Attach the hidden truth as `truth_label` when it exists.
    #
    # The synthetic truth file is exhaustive, so any candidate pair absent from
    # it is genuinely a non-match. That gives a complete, independent labeling of
    # the candidate set -- which is the only way to see how optimistic the
    # silver-label metrics are. `truth_label` is strictly an evaluation column:
    # it is in NON_FEATURE_COLUMNS and never reaches the model.
    if truth is not None and not truth.empty:
        tkey = set(
            zip(truth["source1_id"], truth["candidate_id"], truth["candidate_source"])
        )
        pair_key = list(zip(fdf["s1_id"], fdf["candidate_id"], fdf["candidate_source"]))
        fdf["truth_label"] = [1 if k in tkey else 0 for k in pair_key]
        n_in_truth = int(fdf["truth_label"].sum())
        print(f"  truth labels attached: {n_in_truth:,} true matches among "
              f"{len(fdf):,} candidates ({n_in_truth / max(1, len(fdf)):.2%} positive)")

    # Cache the expensive middle stages so label review can happen offline.
    # Pickle rather than parquet: it is lossless for mixed dtypes (including the
    # `label` column of object dtype holding pd.NA) and needs no extra engine.
    _ensure_dirs(os.path.join(work_dir, "interim"))
    fdf.to_pickle(os.path.join(work_dir, "interim", "features.pkl"))
    for name, frame in cleaned.items():
        frame.to_pickle(os.path.join(work_dir, "interim", f"clean_{name}.pkl"))
    if recall and recall.get("available") and len(recall.get("missed_pairs", [])):
        recall["missed_pairs"].to_csv(
            os.path.join(work_dir, "blocking_missed_pairs.csv"), index=False
        )

    return {
        "s1": s1, "cand_by_source": cand_by_source,
        "s1_id_col": s1_id_col, "cand_id_cols": cand_id_cols,
        "features": fdf, "feature_cols": fcols,
        "block_stats": block_stats, "blocking_recall": recall,
        "full_cross_product": n_full,
    }


# ---------------------------------------------------------------------------
# Stage 5: labels
# ---------------------------------------------------------------------------

def stage_labels(features_df: pd.DataFrame, work_dir: str) -> Dict:
    _ensure_dirs(work_dir)
    print("[5/7] label construction")
    f = labels.build_silver_labels(features_df)
    dist = labels.label_distribution(f)
    print(f"  strata: {dist}")
    n_pos = int(f["silver_label"].eq(1).sum())
    n_neg = int(f["silver_label"].eq(0).sum())
    print(f"  silver labels: {n_pos} match / {n_neg} non-match "
          f"({n_pos / max(1, n_pos + n_neg):.2%} positive)")

    sheet = labels.build_labeling_sheet(f)
    sheet_path = os.path.join(work_dir, "labeling_sheet.csv")
    sheet.to_csv(sheet_path, index=False)
    print(f"  manual labeling sheet: {sheet_path} ({len(sheet)} pairs, label column blank)")

    return {"features": f, "sheet": sheet, "sheet_path": sheet_path, "distribution": dist}


# ---------------------------------------------------------------------------
# Stage 6: train + tune
# ---------------------------------------------------------------------------

def stage_train(
    f: pd.DataFrame,
    work_dir: str,
    feature_cols: List[str],
    label_col: str = "label",
    label_kind: str = "silver",
    target_precision: float = 0.99,
    seed: int = 42,
    suffix: str = "",
) -> Dict:
    _ensure_dirs(work_dir)
    print(f"[6/7] training and threshold tuning  (label_col={label_col}, kind={label_kind})")

    if label_col not in f.columns or f[label_col].notna().sum() == 0:
        f = f.copy()
        f["label"] = f["silver_label"]

    splits = model.make_splits(f, label_col=label_col, seed=seed)
    print(f"  split: {splits.sizes()}  "
          f"positives: train={int(splits.train['label'].sum())} "
          f"val={int(splits.val['label'].sum())} "
          f"test={int(splits.test['label'].sum())}")

    clf = model.MatchClassifier(random_state=seed)
    clf.fit(splits.train, splits.val, feature_cols)
    print(f"  backend: {clf.backend}  (iterations: {clf.best_iteration()})")

    # ---- threshold chosen on validation only ------------------------------
    y_val = splits.val["label"].astype(int).values
    s_val = clf.predict_proba(splits.val)
    sweep = evaluate.sweep_thresholds(y_val, s_val)
    rec = evaluate.recommend_threshold(sweep, target_precision=target_precision)

    chosen = rec.get("precision_target") or rec.get("max_f1")
    threshold = float(chosen["threshold"])
    print("  threshold candidates:")
    for key in ("max_f1", "precision_target"):
        if rec.get(key):
            c = rec[key]
            print(f"    {key:17s} t={c['threshold']:.4f}  "
                  f"P={c['precision']:.4f}  R={c['recall']:.4f}  F1={c['f1']:.4f}  "
                  f"n_pred={c['n_predicted_positive']}")
    print(f"  -> using t={threshold:.4f} (precision_target)")

    val_metrics = evaluate.evaluate_at(y_val, s_val, threshold)

    # ---- test set touched exactly once, here ------------------------------
    y_test = splits.test["label"].astype(int).values
    s_test = clf.predict_proba(splits.test)
    test_metrics = evaluate.evaluate_at(y_test, s_test, threshold)
    print(f"  VALIDATION ({label_kind} labels): P={val_metrics['precision']:.4f} "
          f"R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f}")
    print(f"  TEST       ({label_kind} labels): P={test_metrics['precision']:.4f} "
          f"R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} "
          f"AUC={test_metrics.get('roc_auc')}")

    # ---- the same splits re-scored against the hidden truth ---------------
    # The threshold was picked on silver labels, so this is not a fully clean
    # held-out comparison either, but it is measured against labels the model
    # never saw in any form and is the number that means something.
    truth_metrics = None
    if "truth_label" in splits.test.columns and splits.test["truth_label"].notna().any():
        print("\n  --- re-scored against HIDDEN TRUTH (independent labels) ---")
        truth_val = evaluate.evaluate_at(
            splits.val["truth_label"].astype(int).values, s_val, threshold
        )
        truth_test = evaluate.evaluate_at(
            splits.test["truth_label"].astype(int).values, s_test, threshold
        )
        print(f"  VALIDATION (truth): P={truth_val['precision']:.4f} "
              f"R={truth_val['recall']:.4f} F1={truth_val['f1']:.4f} "
              f"AUC={truth_val.get('roc_auc')}")
        print(f"  TEST       (truth): P={truth_test['precision']:.4f} "
              f"R={truth_test['recall']:.4f} F1={truth_test['f1']:.4f} "
              f"AUC={truth_test.get('roc_auc')}")
        truth_metrics = {"validation": truth_val, "test": truth_test}
        if label_kind == "silver":
            # Reported as the size of the optimism, so a positive number means
            # the silver labels flattered the result.
            gap = test_metrics["precision"] - truth_test["precision"]
            print(f"  optimism of silver labels on test precision: {gap:+.4f} "
                  f"({test_metrics['precision']:.4f} reported vs "
                  f"{truth_test['precision']:.4f} true)")

    if label_kind == "silver":
        print("  *** the ({}) metrics are on RULE-DERIVED labels and are "
              "OPTIMISTIC by construction. ***".format(label_kind))

    plot_path = os.path.join(work_dir, f"threshold_curves{suffix}.png")
    evaluate.make_plots(sweep, y_val, s_val, threshold, plot_path, clf.importances)
    print(f"  plots: {plot_path}")

    clf.importances.to_csv(
        os.path.join(work_dir, f"feature_importance{suffix}.csv"), index=False
    )
    sweep.to_csv(os.path.join(work_dir, f"threshold_sweep{suffix}.csv"), index=False)

    return {
        "classifier": clf,
        "threshold": threshold,
        "sweep": sweep,
        "recommendation": rec,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "truth_metrics": truth_metrics,
        "backend": clf.backend,
        "splits": splits,
        "importances": clf.importances,
        "label_kind": label_kind,
    }


# ---------------------------------------------------------------------------
# Stage 7: score everything and assemble
# ---------------------------------------------------------------------------

def stage_assemble(
    f: pd.DataFrame,
    s1: pd.DataFrame,
    s1_id_col: str,
    clf: model.MatchClassifier,
    feature_cols: List[str],
    threshold: float,
    out_dir: str,
    margin: float = 0.0,
) -> Dict:
    _ensure_dirs(out_dir)
    print("[7/7] scoring all candidates and assembling the match table")

    all_scores = clf.predict_proba(f)
    scored = assemble.build_pair_scores(f, all_scores, threshold)
    if margin > 0:
        # Optional per-entity margin for consumers that cannot tolerate several
        # matches per entity. Kept off by default: many-to-many is legitimate.
        kept = assemble.dedupe_by_margin(scored, threshold, margin=margin)
        scored = scored.copy()
        scored["accepted"] = scored.index.isin(set(kept.index))

    entity_table = assemble.build_entity_table(scored, s1, s1_id_col, threshold)
    summary = assemble.summarize(entity_table, scored, s1, s1_id_col)

    pair_out = scored[scored["accepted"]][
        ["s1_id", "candidate_id", "candidate_source", "score"]
    ].sort_values(["s1_id", "candidate_source", "score"], ascending=[True, True, False])
    pair_out.to_csv(os.path.join(out_dir, "entity_pair_scores.csv"), index=False)
    entity_table.to_csv(os.path.join(out_dir, "entity_matches.csv"), index=False)

    print(f"  {summary['n_pairs_accepted']:,} accepted pairs "
          f"({summary['acceptance_rate']:.2%} of {summary['n_candidate_pairs_scored']:,})")
    for src, s in summary["per_source"].items():
        print(f"    {src}: {s['accepted_pairs']:,} pairs covering "
              f"{s['distinct_source1_entities_matched']:,} Source 1 entities")
    print(f"  entities with no match: {summary['entities_with_zero_matches']:,} "
          f"({summary['entities_with_zero_matches_pct']}%)")
    print(f"  wrote {os.path.join(out_dir, 'entity_matches.csv')}")
    print(f"  wrote {os.path.join(out_dir, 'entity_pair_scores.csv')}")

    review = evaluate.sample_for_manual_review(scored, threshold, n=200)
    review.to_csv(os.path.join(out_dir, "accepted_pairs_to_review.csv"), index=False)
    print(f"  wrote {os.path.join(out_dir, 'accepted_pairs_to_review.csv')} "
          f"({len(review)} accepted pairs for precision spot-check)")

    return {"scored": scored, "entity_table": entity_table, "summary": summary}


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run(
    config_path: str = "config/schema.yaml",
    root: str = ".",
    work_dir: str = "artifacts",
    out_dir: str = "output",
    n_entities: int = 1200,
    generate_data: bool = False,
    seed: int = 42,
    block_cfg: Optional[BlockConfig] = None,
    labels_path: Optional[str] = None,
    target_precision: float = 0.99,
    margin: float = 0.0,
    truth_path: str = "data/_truth/truth_pairs.csv",
    oracle_run: bool = False,
) -> Dict:
    """Run every stage in order and write a run manifest.

    ``oracle_run`` trains directly on the hidden truth file. It exists to prove
    the feature set and the modelling choices are sound, and to show what the
    pipeline achieves once *real* labels exist. It is not a production mode: on
    real data the equivalent is to supply ``labels_path`` with human labels.
    """
    block_cfg = block_cfg or BlockConfig()
    manifest: Dict[str, object] = {"config": config_path, "seed": seed}

    if generate_data:
        print("[1/7] generating synthetic benchmark data")
        manifest["generate"] = stage_generate(
            n_entities, os.path.join(root, "data/raw"),
            os.path.join(root, "data/_truth"), seed
        )
    else:
        print("[1/7] using existing input files")

    prep = stage_prepare(config_path, root, work_dir, block_cfg, truth_path=truth_path)
    manifest["clean"] = {
        "rows": {k: len(v) for k, v in prep["cand_by_source"].items()},
        "n_source1": len(prep["s1"]),
    }
    manifest["block"] = {
        "stats": prep["block_stats"],
        "full_cross_product": prep["full_cross_product"],
        "recall": prep["blocking_recall"],
    }
    manifest["features"] = {
        "n_candidate_pairs": len(prep["features"]),
        "n_features": len(prep["feature_cols"]),
    }

    lab = stage_labels(prep["features"], work_dir)
    manifest["labels"] = {
        "distribution": lab["distribution"],
        "sheet": lab["sheet_path"],
    }

    # Human labels, if supplied, take precedence over silver labels.
    if labels_path:
        print(f"  loading reviewed labels from {labels_path}")
        reviewed = labels.load_reviewed_labels(lab["features"], labels_path)
        n_lab = int(reviewed["label"].notna().sum())
        print(f"  {n_lab} pairs carry a human label "
              f"({int(reviewed['label'].eq(1).sum())} match / "
              f"{int(reviewed['label'].eq(0).sum())} non-match)")
        train_frame, label_kind, label_col = reviewed, "human", "label"
    else:
        train_frame, label_kind, label_col = lab["features"], "silver", "label"

    suffix = ""
    if oracle_run:
        if "truth_label" not in train_frame.columns:
            raise ValueError(
                "--oracle-run needs the hidden truth file "
                f"({truth_path}). It only exists for the generated benchmark."
            )
        train_frame = train_frame.copy()
        train_frame["label"] = train_frame["truth_label"]
        label_kind, label_col = "hidden-truth", "label"
        suffix = "_oracle"
        print("  *** ORACLE RUN: training on hidden truth. Validates the feature "
              "set, but this is not available in production. ***")

    trained = stage_train(
        train_frame, work_dir, prep["feature_cols"],
        label_col=label_col, label_kind=label_kind,
        target_precision=target_precision, seed=seed, suffix=suffix,
    )
    manifest["model"] = {
        "backend": trained["backend"],
        "threshold": trained["threshold"],
        "validation": trained["val_metrics"],
        "test": trained["test_metrics"],
        "truth_validation": (trained["truth_metrics"] or {}).get("validation"),
        "truth_test": (trained["truth_metrics"] or {}).get("test"),
        "recommendation": trained["recommendation"],
        "label_kind": label_kind,
    }
    trained["importances"].head(20).to_csv(
        os.path.join(work_dir, f"top_features{suffix}.csv"), index=False
    )

    result = stage_assemble(
        lab["features"], prep["s1"], prep["s1_id_col"],
        trained["classifier"], prep["feature_cols"],
        trained["threshold"], out_dir, margin=margin,
    )
    manifest["output"] = result["summary"]

    manifest_path = os.path.join(work_dir, "run_manifest.json")
    _write_json(manifest_path, manifest)
    print(f"\n  manifest: {manifest_path}")
    return manifest


# ===========================================================================
# Real-schema entity resolution (entity_id / business_name /
# business_address / country)
#
# `run` above drives the richer synthetic benchmark. `run_er_pipeline` drives
# the real dataset and differs from it in three structural ways:
#
#   1. It streams. Source 1 is indexed once and held; Source 2 and Source 3 are
#      read in chunks and probed against it, so peak memory is "the Source 1
#      index + one chunk + that chunk's pairs" rather than a joined 12.5M-row
#      table. Nothing in the pipeline ever forms a Source1 x Source2/3 product.
#   2. It makes two passes over the candidate stream. The first pass collects a
#      bounded training sample and caches the *tune-split* pair features to
#      disk; the classifier is then fitted, the threshold is tuned on the cached
#      rows, and the second pass reloads the model, decides and writes. The
#      alternative -- keeping 30M scored pairs in memory between fitting and
#      deciding -- is exactly what the chunked design exists to avoid.
#   3. It evaluates with F_0.5 and nothing else.
# ===========================================================================

def _config_section(cls, section: Optional[dict]) -> object:
    """Build a dataclass from a config dict, ignoring unknown keys.

    Unknown keys are ignored rather than fatal so a config can carry comments
    and options this code does not use yet, without pinning the code to them.
    """
    fields = set(getattr(cls, "__dataclass_fields__", {}))
    return cls(**{k: v for k, v in (section or {}).items() if k in fields})


def _stable_bucket(key: str, salt: str = "") -> float:
    """Deterministic value in [0, 1) for a string.

    Python's `hash()` is salted per process, so a split built from it changes
    between runs and the reported number stops being reproducible. md5 does not.
    """
    digest = hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def assign_splits(
    entity_ids: Sequence[str],
    val_fraction: float = 0.20,
    tune_fraction: float = 0.50,
    salt: str = "er-split",
) -> Dict[str, str]:
    """Split Source 1 entities into ``train`` / ``tune`` / ``report``.

    The split is by *Source 1 entity*, not by candidate pair, because the metric
    is per Source 1 entity: two records of the same entity landing on opposite
    sides of the split would leak the answer.

    ``report`` is the held-out set the threshold is never tuned on, so the
    reported F_0.5 comes from data that played no part in choosing it. ``tune``
    exists for that choice, and ``train`` for fitting. Collapsing the first two
    would be the easy way to make the number look better than it is.
    """
    out: Dict[str, str] = {}
    for eid in entity_ids:
        b = _stable_bucket(str(eid), salt)
        if b >= val_fraction:
            out[str(eid)] = "train"
        elif b < val_fraction * tune_fraction:
            out[str(eid)] = "tune"
        else:
            out[str(eid)] = "report"
    return out


def _merge_record_summaries(
    primary: pd.DataFrame, fallback: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Overlay the fallback pass's per-record summary onto the first pass's.

    The fallback pass only touches records the first pass left with zero
    candidates, and it either gives them candidates or leaves them at zero with
    a different reason -- so a row-wise overwrite is exactly the right merge.
    """
    if fallback is None or not len(fallback):
        return primary
    out = primary.set_index("source_record_id")
    # Both frames are built by `block.generate_candidate_pairs` /
    # `fallback_candidate_generation`, which emit identical dtypes (int64
    # counters, object strings) precisely so this overlay is dtype-clean.
    out.update(fallback.set_index("source_record_id"))
    return out.reset_index()


def _feature_columns(frame: pd.DataFrame) -> List[str]:
    """Numeric model inputs, with the label kept out of the matrix."""
    return [c for c in feat.feature_columns(frame) if c != "label"]


def _with_fallback(
    batch: block.CandidateBatch,
    chunk: pd.DataFrame,
    indexes: block.Source1Indexes,
    id_col: str,
    source_dataset: str,
    block_cfg: block.MultiIndexBlockConfig,
    fallback_totals: Dict[str, int],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Candidates for one chunk, with the looser second pass applied.

    Change 4, in one place: the fallback runs on the rows the first pass left
    at zero and nothing else, so it can only ever add candidates. Its thresholds
    are candidate-generation thresholds; every pair it produces still has to
    beat the same F_0.5-tuned classifier threshold as any other pair.
    """
    if not block_cfg.enable_fallback or not len(batch.records):
        return batch.pairs, None
    unresolved = batch.records[batch.records["candidate_count"] == 0]
    if not len(unresolved):
        return batch.pairs, None
    fb = block.fallback_candidate_generation(
        unresolved, chunk, indexes, id_col=id_col,
        source_dataset=source_dataset, cfg=block_cfg,
    )
    for key in fallback_totals:
        fallback_totals[key] = fallback_totals.get(key, 0) + int(fb.stats.get(key, 0))
    if not len(fb.pairs):
        return batch.pairs, fb.records
    return pd.concat([batch.pairs, fb.pairs], ignore_index=True), fb.records


def _pair_labels(
    feats: pd.DataFrame,
    truth: Dict[str, Set[str]],
    split: Dict[str, str],
    wanted: str = "train",
) -> pd.Series:
    """Label each pair 1/0 from the truth file; NaN for split != ``wanted``.

    The truth file is exhaustive -- every true pair of every entity is listed --
    so a candidate pair that is *not* in it is a genuine non-match rather than
    an unknown. That is what makes negative labels trustworthy rather than
    assumed.

    Rows outside ``wanted`` come back NaN, which is how the tune split and the
    report split stay out of the training matrix and out of the threshold sweep.
    """
    n = len(feats)
    if not truth or n == 0:
        return pd.Series(np.nan, index=feats.index)
    s1_ids = feats["s1_entity_id"].astype(str).to_numpy()
    rec_ids = feats["source_record_id"].astype(str).to_numpy()
    y = np.full(n, np.nan)
    empty: Set[str] = set()
    for i in range(n):
        side = split.get(s1_ids[i])
        if side != wanted:
            continue
        y[i] = 1.0 if rec_ids[i] in truth.get(s1_ids[i], empty) else 0.0
    return pd.Series(y, index=feats.index)


def run_er_pipeline(
    config_path: str = "config/er.yaml",
    root: str = ".",
    out_dir: str = "output",
    work_dir: str = "artifacts/er",
    mode: str = "validate",
    model_path: Optional[str] = None,
    max_source1: Optional[int] = None,
    max_source2: Optional[int] = None,
    max_source3: Optional[int] = None,
    max_chunks: Optional[int] = None,
    seed: int = 42,
    log: Optional[logging.Logger] = None,
) -> Dict:
    """Run blocking -> features -> classifier -> tiered decision -> outputs.

    ``mode="validate"``
        Train against the ground-truth file. The F_0.5-maximizing
        ``high_threshold`` is tuned on the tune split, and the held-out report
        split produces the single reported F_0.5.
    ``mode="predict"``
        No labels. Requires ``model_path`` (a directory written by a previous
        validate run) and takes the thresholds from the config. F_0.5 is *not*
        computed, because there is nothing to compute it against -- see
        `evaluate.final_evaluation`.

    The row caps are development shortcuts: they subsample the sources, so any
    number produced with them set describes the subsample, not the dataset.
    """
    log = log or logging.getLogger("pipeline.er")
    started = time.time()
    config = data.load_config(config_path)
    _ensure_dirs(work_dir, out_dir)

    paths = (config.get("paths") or {}).get(mode) or {}
    sources = config.get("sources") or {}
    sep = config.get("sep", "\t")
    chunk_size = int(config.get("chunk_size", 50_000))
    block_cfg = _config_section(block.MultiIndexBlockConfig, config.get("blocking"))
    weights = feat.CompositeWeights(**(config.get("weights") or {}))
    val_cfg = config.get("validation") or {}
    val_fraction = float(val_cfg.get("val_fraction", 0.20))
    tune_fraction = float(val_cfg.get("tune_fraction", 0.50))
    max_training_pairs = int(config.get("max_training_pairs", 500_000))
    max_tuning_pairs = int(config.get("max_tuning_pairs", 2_000_000))
    hard_cfg = hardneg.HardNegConfig(**(config.get("hardneg") or {}))
    if hard_cfg.enabled:
        log.info("  hard negatives: ON (hard=%.2f locality=%.2f random=%.2f cap/entity=%d)",
                 hard_cfg.hard_fraction, hard_cfg.locality_fraction,
                 hard_cfg.random_fraction, hard_cfg.max_per_entity)
    model_cfg = config.get("model") or {}
    decision_overrides = dict(config.get("decision") or {})

    # ---- Stage 1: Source 1 and its indexes ---------------------------------
    log.info("[1/5] indexing Source 1")
    s1_spec = sources.get("source1") or {}
    s1 = data.load_entity_source(
        paths.get("source1", ""), s1_spec, root=root, sep=sep, nrows=max_source1
    )
    id_col = str(s1_spec.get("id_col") or "entity_id")
    all_s1_ids: List[str] = [str(v) for v in s1[id_col].tolist()]

    # The submission needs exactly one row per Source 1 record in the *file*,
    # not per record that survived loading. If a row cap is still set, `s1` is
    # short, every downstream count silently describes a subsample, and the
    # output is a truncated submission that looks well-formed.
    _s1_path = paths.get("source1")
    if _s1_path:
        _n_file = 0
        with open(_s1_path, "r", encoding="utf-8") as _fh:
            next(_fh, None)
            for _line in _fh:
                if _line.strip():
                    _n_file += 1
        if _n_file != len(all_s1_ids):
            raise AssertionError(
                f"loaded {len(all_s1_ids):,} Source 1 records but {_s1_path} has "
                f"{_n_file:,}. A row cap is truncating the input; the submission "
                f"needs one row per Source 1 entity."
            )
        log.info("  Source 1 coverage: %s loaded == %s in file",
                 f"{len(all_s1_ids):,}", f"{_n_file:,}")
    indexes = block.build_source1_indexes(s1, block_cfg, id_col=id_col)
    block.attach_source1_frame(indexes, s1)
    log.info("  %d Source 1 entities indexed in %.1fs | buckets: %s",
             len(s1), time.time() - started, indexes.stats.get("bucket_counts"))

    truth: Dict[str, Set[str]] = {}
    split: Dict[str, str] = {}
    if mode == "validate":
        truth = data.load_ground_truth_map(paths.get("ground_truth", ""), root=root)
        split = assign_splits(all_s1_ids, val_fraction, tune_fraction)
        log.info("  entity split: %s",
                 pd.Series(list(split.values())).value_counts().to_dict())

    def iter_chunks(src: str) -> Iterator[pd.DataFrame]:
        cap = max_source2 if src == "source2" else max_source3
        return data.iter_entity_chunks(
            paths.get(src, ""), sources.get(src) or {}, root=root, sep=sep,
            chunksize=chunk_size, nrows=cap,
        )

    fallback_totals: Dict[str, int] = {
        "n_input_unresolved": 0, "n_rescued": 0, "n_still_unresolved": 0,
    }
    block_stats: Dict[str, object] = dict(indexes.stats)
    clf: Optional[model.MatchClassifier] = None
    feature_cols: List[str] = []

    # ---- Pass 1: training sample + cached tune-split features --------------
    if mode == "validate":
        log.info("[2/5] pass 1 -- collecting training pairs")
        train, tune_cache = _collect_training_sample(
            s1, indexes, block_cfg, id_col, iter_chunks, weights, truth, split,
            max_training_pairs, max_tuning_pairs, seed, fallback_totals, log,
            hard_cfg,
        )
        if train is None or not len(train):
            raise RuntimeError(
                "no labeled candidate pairs were generated. Check that the "
                "config paths point at the right files, that the id column is "
                "right, and that the ground truth matches the Source 1 entities."
            )
        feature_cols = _feature_columns(train)
        log.info("  training pairs: %d (positives %d) in %.1fs",
                 len(train), int(train["label"].sum()), time.time() - started)

        splits = model.make_splits(train, label_col="label", seed=seed)
        clf = model.MatchClassifier(
            **_classifier_kwargs(model_cfg, seed)
        )
        clf.fit(splits.train, splits.val, feature_cols)
        log.info("  backend: %s (iterations: %s)", clf.backend, clf.best_iteration())
        clf.save(os.path.join(work_dir, "model"))

        # Threshold tuning uses the cached tune-split rows, so the reported
        # number comes from a split that had no influence on either the model or
        # the threshold.
        log.info("[3/5] tuning the decision threshold on the tune split")
        if tune_cache is not None and len(tune_cache):
            y = _pair_labels(tune_cache, truth, split, wanted="tune").to_numpy()
            keep = ~np.isnan(y)
            scores = clf.predict_proba(tune_cache.loc[keep])
            tuning, sweep = evaluate.tune_threshold(
                y[keep].astype(int), scores, return_sweep=True
            )
            sweep.to_csv(os.path.join(work_dir, "threshold_sweep.csv"), index=False)
        else:
            tuning, sweep = {"high_threshold": float(decision_overrides.get(
                "high_threshold", 0.85)), "f05": 0.0, "n_thresholds": 0}, None
        log.info(
            "  tuned high_threshold = %.2f (%s on the tune split = %.4f)",
            tuning["high_threshold"], evaluate.F05_LABEL, tuning.get("f05", float("nan")),
        )
    else:
        if not model_path:
            raise ValueError(
                "mode='predict' needs model_path: F_0.5 cannot be computed "
                "without labels, so both thresholds have to come from a "
                "previous validate run"
            )
        log.info("[2/5] loading the trained model from %s", model_path)
        clf = model.MatchClassifier.load(model_path)
        feature_cols = list(clf.feature_cols)
        tuning = {
            "high_threshold": float(decision_overrides.get("high_threshold", 0.85)),
            "note": "carried over from the validate run; not tuned here",
        }
        log.info("[3/5] using high_threshold = %.2f from config",
                 tuning["high_threshold"])

    decision_cfg = _decision_config(decision_overrides, tuning["high_threshold"], log)

    # ---- Pass 2: score, decide, write --------------------------------------
    log.info("[4/5] pass 2 -- scoring candidates and applying the tiered decision")
    accumulator = assemble.RecordAccumulator(
        out_dir, max_rollup_pairs=int(config.get("max_rollup_pairs", 40_000_000))
    )
    debug_frames: List[pd.DataFrame] = []
    n_chunks = 0
    for src in ("source2", "source3"):
        if max_chunks is not None and n_chunks >= max_chunks:
            break
        for chunk in iter_chunks(src):
            if max_chunks is not None and n_chunks >= max_chunks:
                break
            n_chunks += 1
            batch = block.generate_candidate_pairs(
                chunk, indexes, id_col, src, block_cfg
            )
            pairs, fb_records = _with_fallback(
                batch, chunk, indexes, id_col, src, block_cfg, fallback_totals
            )
            if len(batch.debug):
                debug_frames.append(batch.debug)
            records = _merge_record_summaries(batch.records, fb_records)
            if len(pairs):
                feats = _score_pairs(pairs, chunk, s1, id_col, weights)
                if len(feats):
                    missing = [c for c in feature_cols if c not in feats.columns]
                    if missing:
                        raise RuntimeError(
                            f"feature frame is missing model columns: {missing[:5]}"
                        )
                    feats["classifier_probability"] = clf.predict_proba(feats)
                    record_table = assemble.build_record_table(
                        assemble.decide_match_status(feats, decision_cfg),
                        records, feats,
                    )
                    accumulator.add(record_table, feats)
                    continue
            # No pairs at all for this chunk: the records still have to appear
            # in the output, as unmatched with the reason recorded.
            accumulator.add(
                assemble.build_record_table(
                    pd.DataFrame(columns=["source_record_id", "match_status",
                                          "decision_notes", "source_dataset"]),
                    records,
                    pd.DataFrame(),
                )
            )
    counters = accumulator.close()
    log.info("  pass 2 done in %.1fs", time.time() - started)

    # ---- the one reported number ------------------------------------------
    validation: Dict[str, object] = {
        "high_threshold": decision_cfg.high_threshold,
        "low_threshold": decision_cfg.low_threshold,
        "ambiguity_margin": decision_cfg.ambiguity_margin,
        "threshold_tuning": {
            "split": "tune" if mode == "validate" else None,
            "high_threshold": tuning["high_threshold"],
            evaluate.F05_LABEL: tuning.get("f05"),
        },
    }
    if mode == "validate":
        report_ids = [e for e in all_s1_ids if split.get(e) == "report"]
        truth_report = {e: truth.get(e, set()) for e in report_ids}
        predicted_report = {
            k: v for k, v in accumulator.matched_by_entity.items()
            if split.get(k) == "report"
        }
        # Held-out report split: the threshold was tuned on `tune`, so this is
        # the only F_0.5 in the run that is measured on data which influenced
        # neither the model nor the threshold.
        validation.update(
            evaluate.final_evaluation_macro(
                predicted_report, truth_report, report_ids, label="held-out validation"
            )
        )
        # Blocking recall: the fraction of true pairs the candidate stage ever
        # produced. Not a model metric -- it is the ceiling on recall, and it is
        # the first thing to check when F_0.5 disappoints.
        validation["blocking_recall"] = _blocking_recall(
            accumulator, truth, report_ids
        )
    else:
        validation["note"] = (
            "F_0.5 is not computed on unlabeled data; it requires true labels. "
            "Only the thresholds carried over from the validate run are set."
        )

    summary = assemble.summarize_er_run(
        counters, block_stats, fallback_totals,
        {
            "mode": mode,
            "chunk_size": chunk_size,
            "blocking": _as_dict(block_cfg),
            "weights": weights.as_dict(),
            "decision": _as_dict(decision_cfg),
            "model_backend": getattr(clf, "backend", None),
            "n_source1_rows": len(s1),
            "row_caps": {
                "max_source1": max_source1, "max_source2": max_source2,
                "max_source3": max_source3, "max_chunks": max_chunks,
            },
        },
        validation=validation,
    )
    summary["rollup_saturated"] = accumulator.rollup_saturated

    log.info("[5/5] writing outputs to %s", out_dir)
    debug = pd.concat(debug_frames, ignore_index=True) if debug_frames else pd.DataFrame()
    written = assemble.write_er_outputs(
        accumulator, debug, all_s1_ids, out_dir, summary
    )
    return {
        "outputs": written,
        "summary": summary,
        "model_backend": getattr(clf, "backend", None),
        "feature_columns": feature_cols,
    }


def _classifier_kwargs(model_cfg: Dict[str, object], seed: int) -> Dict[str, object]:
    """Whitelisted `MatchClassifier` kwargs from the config section."""
    allowed = ("prefer", "n_estimators", "learning_rate", "max_leaf_nodes",
               "min_samples_leaf", "n_jobs", "random_state")
    kwargs = {k: v for k, v in (model_cfg or {}).items() if k in allowed}
    kwargs.setdefault("random_state", seed)
    return kwargs


def _as_dict(obj: object) -> Dict[str, object]:
    return dict(vars(obj)) if hasattr(obj, "__dict__") else {}


def _decision_config(
    overrides: Dict[str, object], high: float, log: logging.Logger
) -> assemble.DecisionConfig:
    """Resolve the two thresholds against the F_0.5-tuned ``high``.

    ``low_threshold_ratio`` (default 0.6) expresses the floor as a fraction of
    the tuned high threshold, so the middle tier keeps its width when the high
    threshold moves. An absolute ``low_threshold`` in the config is still
    honoured as a cap, but never above the tuned high: a ``low`` above ``high``
    would make the manual-review tier unreachable and quietly turn every
    uncertain record into either a match or nothing at all.
    """
    cfg = dict(overrides or {})
    ratio = cfg.pop("low_threshold_ratio", 0.6)
    cfg["high_threshold"] = float(high)
    low = cfg.get("low_threshold")
    if low is None:
        low = round(float(ratio) * float(high), 4)
    else:
        low = min(float(low), round(float(ratio) * float(high), 4))
    if low > float(high):
        log.warning(
            "low_threshold %.3f exceeds the tuned high_threshold %.3f; clamping "
            "it (the manual_review tier would otherwise be unreachable)",
            low, float(high),
        )
        low = float(high)
    cfg["low_threshold"] = max(0.0, low)
    return _config_section(assemble.DecisionConfig, cfg)


def _collect_training_sample(
    s1: pd.DataFrame,
    indexes: block.Source1Indexes,
    block_cfg: block.MultiIndexBlockConfig,
    id_col: str,
    iter_chunks: object,
    weights: object,
    truth: Dict[str, Set[str]],
    split: Dict[str, str],
    max_training_pairs: int,
    max_tuning_pairs: int,
    seed: int,
    fallback_totals: Dict[str, int],
    log: logging.Logger,
    hard_cfg: Optional['hardneg.HardNegConfig'] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Pass 1: a bounded labelled training matrix and a tune-split cache.

    Every positive pair is kept -- they are the scarce class and the only thing
    standing between the model and a degenerate all-negative classifier. Negatives
    are sampled up to the budget, which is safe because at ~1% positive rate a
    random negative sample is representative and the alternative is a training
    set dominated by pairs that are trivially separable anyway.

    The tune-split rows are cached (capped, then thinned with a deterministic
    stride) so the threshold can be tuned *after* the model exists without a
    second pass over the whole candidate stream.
    """
    hard_cfg = hard_cfg or hardneg.HardNegConfig()
    pos_frames: List[pd.DataFrame] = []
    neg_frames: List[pd.DataFrame] = []
    tune_frames: List[pd.DataFrame] = []
    n_pos = 0
    n_neg = 0
    tune_rows = 0

    for src in ("source2", "source3"):
        for chunk in iter_chunks(src):  # type: ignore[union-attr]
            batch = block.generate_candidate_pairs(
                chunk, indexes, id_col, src, block_cfg
            )
            pairs, _ = _with_fallback(batch, chunk, indexes, id_col, src,
                                      block_cfg, fallback_totals)
            if not len(pairs):
                continue
            feats = _score_pairs(pairs, chunk, s1, id_col, weights)
            if not len(feats):
                continue
            cols = _feature_columns(feats)

            # Cached for threshold tuning: the tune split, which is never
            # trained on.
            if tune_rows < max_tuning_pairs:
                side = feats["s1_entity_id"].astype(str).map(split)
                sel = feats.loc[side == "tune", cols + ["s1_entity_id",
                                                          "source_record_id"]]
                if len(sel):
                    tune_frames.append(sel)
                    tune_rows += len(sel)

            y = _pair_labels(feats, truth, split, wanted="train")
            labeled = y.notna()
            if not labeled.any():
                continue
            y = y[labeled].astype(int)
            frame = feats.loc[labeled, cols]
            positives = frame[y.to_numpy() == 1]
            negatives = frame[y.to_numpy() == 0]
            n_pos += len(positives)
            n_neg += len(negatives)
            if len(positives):
                pos_frames.append(positives)
            room = max(0, max_training_pairs - n_pos)
            if room and len(negatives):
                # Hard-negative-aware selection. Only train-split entities reach
                # this point (the label gate above returns NaN for tune/report),
                # so the tune cache and the held-out report split are untouched
                # and the reported score stays comparable to the baseline.
                sel = hardneg.select_hard_negatives(
                    negatives,
                    hard_cfg,
                    budget=min(room, len(negatives)),
                    seed=seed + len(neg_frames),
                    truth=truth,
                )
                if len(sel):
                    neg_frames.append(sel)
            log.info("  %s: pairs so far %d (pos %d)", src, n_pos + n_neg, n_pos)
            if n_pos >= max_training_pairs:
                break
        if n_pos >= max_training_pairs:
            break

    frames = pos_frames + neg_frames
    train = None
    if frames:
        # Label vector assembled alongside the frames: positives first, then the
        # sampled negatives, in the order `pd.concat` will produce them.
        labels = np.concatenate(
            [np.ones(len(f), dtype=int) for f in pos_frames]
            + [np.zeros(len(f), dtype=int) for f in neg_frames]
        )
        train = pd.concat(frames, ignore_index=True)
        train["label"] = labels

    tune_cache = pd.concat(tune_frames, ignore_index=True) if tune_frames else None
    if tune_cache is not None and len(tune_cache) > max_tuning_pairs:
        # Deterministic stride: unbiased, and reproducible run to run.
        stride = int(np.ceil(len(tune_cache) / max_tuning_pairs))
        tune_cache = tune_cache.iloc[::stride].reset_index(drop=True)
    return train, tune_cache


def _score_pairs(
    pairs: pd.DataFrame,
    chunk: pd.DataFrame,
    s1: pd.DataFrame,
    id_col: str,
    weights: object,
) -> pd.DataFrame:
    """Features plus the explainable weighted score for one chunk's pairs."""
    feats = feat.compute_pairwise_features(s1, chunk, pairs, id_col, id_col)
    if not len(feats):
        return feats
    scores, breakdown = feat.compute_weighted_composite_score(
        feats, weights, return_breakdown=True
    )
    feats["weighted_composite_score"] = scores
    for col in breakdown.columns:
        if col != "weighted_composite_score":
            feats[col] = breakdown[col]
    return feats


def _blocking_recall(
    accumulator: assemble.RecordAccumulator,
    truth: Dict[str, Set[str]],
    entity_ids: Sequence[str],
) -> Dict[str, object]:
    """Fraction of true pairs that blocking generated at all.

    A candidate stage that never proposed a true pair cannot be recovered by any
    amount of modelling, so this ceiling is reported next to F_0.5: a low F_0.5
    with high blocking recall is a model problem, and a low F_0.5 with low
    blocking recall is a Stage 1 problem.
    """
    n_true = 0
    n_in_candidates = 0
    for entity_id in entity_ids:
        true_ids = truth.get(entity_id)
        if not true_ids:
            continue
        n_true += len(true_ids)
        n_in_candidates += len(true_ids & accumulator.cand_by_entity.get(entity_id, set()))
    return {
        "n_true_pairs": int(n_true),
        "n_true_pairs_in_candidate_set": int(n_in_candidates),
        "recall": round(n_in_candidates / n_true, 4) if n_true else None,
    }
