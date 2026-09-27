"""Build a labeled subsample of the training data and measure blocking recall.

This is the measurement that decides whether the pipeline is worth running at
full scale, so it runs on a real (if small) slice of the actual challenge data
rather than on synthetic fixtures.

Method: sample N Source 1 entities, keep every Source 2/3 record that the ground
truth says matches one of them (the positives), plus a random sample of records
that match nothing sampled (realistic hard negatives drawn from the same
country mix). That yields a labeled candidate universe small enough to iterate
on, with the same class balance and the same name/address corruption as the real
thing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blocking import BlockingConfig
from src.data_io import build_record_frame
from src.loader import _read_source, run_blocking
from src.metrics import load_ground_truth

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "student_resource", "dataset", "train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=60000)
    ap.add_argument("--n-neg", type=int, default=600000,
                    help="non-matching S2/S3 records to retain as candidates")
    ap.add_argument("--out", default="artifacts/subsample")
    ap.add_argument("--per-key-cap", type=int, default=8)
    ap.add_argument("--bucket-cap", type=int, default=400)
    ap.add_argument("--entity-cap", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print("reading ground truth ...")
    truth, order = load_ground_truth(os.path.join(DATA, "train_ground_truth.tsv"))
    print(f"  {len(truth):,} entities, {sum(len(v) for v in truth.values()):,} true matches "
          f"({time.time() - t0:.0f}s)")

    # ---- sample Source 1 entities ----------------------------------------
    keep_idx = np.sort(rng.choice(len(order), min(args.n_s1, len(order)), replace=False))
    s1_ids = [order[i] for i in keep_idx]
    s1_id_set = set(s1_ids)
    print(f"sampled {len(s1_ids):,} Source 1 entities")

    # ---- the records that are true matches of those entities --------------
    want: Set[str] = set()
    for eid in s1_ids:
        want |= truth[eid]
    print(f"their true matches: {len(want):,}")

    s1 = _read_source(os.path.join(DATA, "train_source1.tsv"))
    s1 = s1[s1.entity_id.isin(s1_id_set)].reset_index(drop=True)

    cand_frames = {}
    for src in (2, 3):
        df = _read_source(os.path.join(DATA, f"train_source{src}.tsv"))
        pos = df[df.entity_id.isin(want)]
        # Hard negatives: records that are true matches of *some other* entity
        # that we did not sample. Taking true matches of unsampled entities is
        # the right negative pool -- they are real records with the same noise
        # profile, not synthetic noise.
        used = want
        rest = df[~df.entity_id.isin(used)]
        take = min(args.n_neg // 2, len(rest))
        neg = rest.iloc[rng.choice(len(rest), take, replace=False)]
        cand_frames[src] = pd.concat([pos, neg], ignore_index=True)
        print(f"  source{src}: {len(pos):,} positives + {len(neg):,} negatives")

    # ---- normalize -------------------------------------------------------
    t0 = time.time()
    frames = []
    recs = {}
    rec = build_record_frame(s1); rec["src"] = np.int8(1)
    frames.append(rec); recs[1] = s1
    for src in (2, 3):
        rec = build_record_frame(cand_frames[src]); rec["src"] = np.int8(src)
        frames.append(rec); recs[src] = cand_frames[src]
    table = pd.concat(frames, ignore_index=True, copy=False)
    print(f"normalized {len(table):,} records in {time.time() - t0:.1f}s")

    counts = {1: len(s1), 2: len(cand_frames[2]), 3: len(cand_frames[3])}

    # ---- blocking --------------------------------------------------------
    cfg = BlockingConfig(per_key_cap=args.per_key_cap, bucket_cap=args.bucket_cap,
                         per_entity_cap=args.entity_cap)
    s1r, candr, stats = run_blocking(table, counts, cfg)

    # ---- blocking recall -------------------------------------------------
    # Map positions -> entity ids so recall is measured against the truth file.
    cand_eid = np.concatenate([
        cand_frames[2].entity_id.to_numpy().astype(object),
        cand_frames[3].entity_id.to_numpy().astype(object),
    ])
    s1_pos_of = {e: i for i, e in enumerate(s1.entity_id)}

    got = set()
    for a, b in zip(s1r, candr):
        got.add((int(a), str(cand_eid[int(b)])))

    n_truth = 0
    n_hit = 0
    missed: List[Dict[str, str]] = []
    per_source = {}
    for eid in s1_ids:
        p = s1_pos_of[eid]
        for m in truth[eid]:
            n_truth += 1
            src = m.split("-")[0].lower()
            d = per_source.setdefault(src, [0, 0])
            d[0] += 1
            if (p, m) in got:
                n_hit += 1
                d[1] += 1
            elif len(missed) < 5000:
                missed.append({"source1_entity_id": eid, "candidate_entity_id": m})

    print("\n=== BLOCKING RECALL ===")
    print(f"  true pairs        : {n_truth:,}")
    print(f"  survived blocking : {n_hit:,}")
    print(f"  RECALL            : {n_hit / max(1, n_truth):.4f}")
    for k, (t, h) in sorted(per_source.items()):
        print(f"    {k}: {h / max(1, t):.4f}  ({h:,}/{t:,})")
    print(f"  candidates/S1     : {stats['mean_candidates_per_s1']:.1f}")
    print(f"  reduction         : {stats['reduction_ratio']:.6%}")

    # ---- persist ---------------------------------------------------------
    # `got` already contains every (s1_row, candidate_entity_id) that survived
    # blocking, which is exactly the positive-label lookup. Building the truth
    # set inside the loop instead would rebuild 138k tuples per candidate pair.
    labels = np.array(
        [1 if (int(a), str(cand_eid[int(b)])) in got else 0
         for a, b in zip(s1r, candr)],
        dtype=np.int8,
    )
    print(f"  labeled {len(labels):,} pairs -> {int(labels.sum()):,} positive "
          f"({labels.mean():.2%})")

    np.savez(os.path.join(args.out, "pairs.npz"), s1r=s1r, candr=candr, label=labels)
    table.to_parquet(os.path.join(args.out, "records.parquet"), index=False)
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump({
            "counts": counts, "stats": {k: v for k, v in stats.items()},
            "blocking_recall": n_hit / max(1, n_truth),
            "n_truth": n_truth, "n_hit": n_hit,
            "per_source_recall": {k: v[1] / max(1, v[0]) for k, v in per_source.items()},
            "n_pairs": int(len(s1r)), "n_positive_pairs": int(labels.sum()),
        }, fh, indent=2, default=str)
    pd.DataFrame(missed).to_csv(os.path.join(args.out, "missed_pairs.csv"), index=False)
    print(f"\nwrote {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
