"""Score a completed validate run: recall, precision, F_0.5, pairs, runtime.

`summary_report.json` reports blocking recall and F_0.5 but **not precision**, so
precision is computed here directly from the emitted match decisions against the
ground truth.

Two precisions are reported and they answer different questions:

  pair precision
      Over emitted (S1, S2/S3) pairs only. This is the number that matters for
      the "merging two real businesses is expensive" concern.

  macro F_0.5
      The competition metric: per-S1-entity F_0.5, averaged over all Source 1
      entities including singletons. Recall on singletons is 0/0, so pair
      precision and F_0.5 can move in opposite directions -- a run can raise
      pair precision while losing more than it gains on singleton entities.
      Reporting only the first would hide exactly the trade the metric rewards.

Blocking recall is also recomputed here against the *achievable* ceiling, because
when the candidate sources are row-capped the reported recall can be pinned by
arithmetic rather than by the blocker.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import f05_macro, load_ground_truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--full-source2", type=int, default=5034616)
    ap.add_argument("--full-source3", type=int, default=5285603)
    ap.add_argument("--sampled-source2", type=int, default=None)
    ap.add_argument("--sampled-source3", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()

    report_path = os.path.join(args.run_dir, "summary_report.json")
    with open(report_path) as fh:
        report = json.load(fh)

    truth, order = load_ground_truth(args.ground_truth)
    n1 = report["validation"].get("n_entities_scored", 0)

    # ---- emitted pairs -> precision -------------------------------------
    matched_path = os.path.join(args.run_dir, "matched_records.csv")
    emitted: Set[tuple] = set()
    if os.path.exists(matched_path):
        import csv

        with open(matched_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("match_status") or "").strip() != "matched":
                    continue
                s1 = (row.get("matched_source1_id") or "").strip()
                cand = (row.get("source_record_id") or "").strip()
                if s1 and cand:
                    emitted.add((s1, cand))

    tp = sum(1 for pair in emitted if pair[1] in truth.get(pair[0], set()))
    fp = len(emitted) - tp
    pair_precision = tp / len(emitted) if emitted else 0.0

    # ---- macro F_0.5 from the emitted matching_results.tsv ---------------
    # Scored over the entities the run actually covered. When the sources are
    # row-capped, matching_results.tsv holds only the sampled Source 1 entities,
    # and scoring against the full 2.2M-entity truth would score every unseen
    # entity 0.0 for the crime of not existing in the output, producing a number
    # dominated by the cap rather than by the model.
    res_path = os.path.join(args.run_dir, "matching_results.tsv")
    predicted: Dict[str, Set[str]] = {}
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if not parts or not parts[0]:
                    continue
                ids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
                predicted[parts[0].strip()] = {x.strip() for x in ids if x.strip()}

    capped = len(predicted) < 0.95 * len(truth)
    scored_over = list(predicted.keys()) if capped else list(truth.keys())
    macro = f05_macro(predicted, truth, all_entities=scored_over)
    restricted = {e: truth.get(e, set()) for e in scored_over}
    pair_recall = tp / sum(len(v) for v in restricted.values()) if restricted else 0.0

    # ---- blocking recall + the ceiling it is measured against -----------
    blocking = report["validation"].get("blocking_recall", {})
    n_true = blocking.get("n_true_pairs", 0)
    found = blocking.get("n_true_pairs_in_candidate_set", 0)
    recall = blocking.get("recall")

    ceiling = None
    if args.sampled_source2 and args.sampled_source3:
        frac = (args.sampled_source2 + args.sampled_source3) / float(
            args.full_source2 + args.full_source3
        )
        ceiling = {"pool_fraction": round(frac, 5), "expected_present": int(n_true * frac),
                   "recall_ceiling": round(frac, 5)}

    out = {
        "recall": recall,
        "precision": round(pair_precision, 4),
        "f0.5": round(macro["f05_macro"], 4),
        "f0.5_reported_by_run": report["validation"].get("F_0.5"),
        "candidate_pairs_scored": report.get("n_candidate_pairs_scored"),
        "avg_candidates_per_entity": report.get("avg_candidate_count"),
        "training_pairs": report.get("n_training_pairs"),
        "emitted_pairs": len(emitted),
        "true_positives": tp,
        "false_positives": fp,
        "pair_recall_over_scored_entities": round(pair_recall, 4),
        "scored_over_predicted_entities_only": capped,
        "pct_entities_perfect": round(macro["pct_perfect"], 2),
        "n_entities_scored": macro["n_entities"],
        "backend": (report.get("config") or {}).get("model_backend"),
        "high_threshold": (report.get("config") or {}).get("decision", {}).get("high_threshold"),
        "runtime_seconds": args.seconds,
        "blocking_recall_detail": blocking,
        "blocking_recall_ceiling": ceiling,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
