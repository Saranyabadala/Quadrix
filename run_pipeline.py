#!/usr/bin/env python
"""Entity resolution across three noisy business-record sources.

Two pipelines live here, selected with ``--mode``:

    # real schema (entity_id / business_name / business_address / country)
    python run_pipeline.py --mode er --er-mode validate
    python run_pipeline.py --mode er --er-mode predict --model-path artifacts/er/model

    # richer synthetic benchmark (the original pipeline)
    python run_pipeline.py --mode benchmark --generate

Outputs, real-schema mode
    output/matched_records.csv         auto-matched Source 2/3 records
    output/manual_review_records.csv   manual_review + ambiguous records
    output/unmatched_records.csv       decided non-matches
    output/unresolved_records.csv      no candidates even after the fallback pass
    output/candidate_debug_log.csv     blocking diagnostics
    output/summary_report.json         counters + the single reported F_0.5
    output/matching_results.tsv        submission: one row per Source 1 entity
    output/candidate_pairs.tsv         submission: candidates per Source 1 entity

Outputs, benchmark mode
    output/entity_matches.csv          source1_id, matched_source2_ids, ...
    output/entity_pair_scores.csv      the audit trail behind the table above
    artifacts/run_manifest.json        every number quoted in the summary
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.block import BlockConfig
from src.pipeline import run, run_er_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Supervised entity resolution for multi-source business records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=("er", "benchmark"), default="er",
                   help="'er' = real 4-column schema; 'benchmark' = synthetic "
                        "rich-schema benchmark")
    p.add_argument("--root", default=".",
                   help="root that source paths in the config are relative to")
    p.add_argument("--work-dir", default=None,
                   help="intermediate outputs (default: artifacts/er or artifacts)")
    p.add_argument("--out-dir", default="output",
                   help="final match tables")

    # --- real-schema (er) options ------------------------------------------
    p.add_argument("--er-config", default="config/er.yaml",
                   help="paths, schema map, weights, thresholds, chunk size")
    p.add_argument("--er-mode", choices=("validate", "predict"), default="validate",
                   help="validate: train, tune the threshold and report F_0.5. "
                        "predict: score unlabeled data with a saved model")
    p.add_argument("--model-path", default=None,
                   help="saved model directory, required by --er-mode predict")
    p.add_argument("--max-source1", type=int, default=None,
                   help="cap Source 1 rows (development subsample)")
    p.add_argument("--max-source2", type=int, default=None,
                   help="cap Source 2 rows (development subsample)")
    p.add_argument("--max-source3", type=int, default=None,
                   help="cap Source 3 rows (development subsample)")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="stop after this many chunks per source")
    p.add_argument("--log-level", default="INFO",
                   help="DEBUG, INFO, WARNING, ...")

    # --- benchmark options -------------------------------------------------
    p.add_argument("--config", default="config/schema.yaml",
                   help="column mapping for the three sources (benchmark mode)")
    p.add_argument("--generate", action="store_true",
                   help="generate the synthetic benchmark instead of reading CSVs")
    p.add_argument("--n-entities", type=int, default=1200,
                   help="Source 1 entities to synthesize (with --generate)")
    p.add_argument("--labels", default=None,
                   help="reviewed labeling sheet; human labels override silver labels")
    p.add_argument("--oracle-run", action="store_true",
                   help="BENCHMARK ONLY: train on the hidden truth file to validate "
                        "the feature set. Never available on real data.")
    p.add_argument("--target-precision", type=float, default=0.99,
                   help="precision floor for the benchmark path's threshold")
    p.add_argument("--margin", type=float, default=0.0,
                   help="per-entity score margin; 0 keeps every pair above threshold")
    p.add_argument("--max-block-size", type=int, default=40,
                   help="skip a blocking key shared by more than this many records")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(message)s",
    )

    if args.mode == "er":
        run_er_pipeline(
            config_path=args.er_config,
            root=args.root,
            out_dir=args.out_dir,
            work_dir=args.work_dir or "artifacts/er",
            mode=args.er_mode,
            model_path=args.model_path,
            max_source1=args.max_source1,
            max_source2=args.max_source2,
            max_source3=args.max_source3,
            max_chunks=args.max_chunks,
            seed=args.seed,
        )
        return 0

    block_cfg = BlockConfig(max_block_size=args.max_block_size)
    run(
        config_path=args.config,
        root=args.root,
        work_dir=args.work_dir or "artifacts",
        out_dir=args.out_dir,
        n_entities=args.n_entities,
        generate_data=args.generate,
        seed=args.seed,
        block_cfg=block_cfg,
        labels_path=args.labels,
        target_precision=args.target_precision,
        margin=args.margin,
        oracle_run=args.oracle_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
