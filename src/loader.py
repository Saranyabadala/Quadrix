"""Loading, normalizing and blocking the challenge data, at scale.

Memory strategy, because 11.7M test records do not fit comfortably as pandas
object strings:

* Sources are read one at a time and immediately reduced to the compact record
  table (`build_record_frame`), which holds int32 codes and a handful of small
  ints per record. The raw text is dropped as soon as it is normalized.
* ``entity_id`` is kept as a pandas Categorical, so 11.7M ids cost a code array
  plus one dictionary of distinct strings rather than 11.7M Python objects.
* Peak resident set for the test split is a few GB, dominated by the transient
  raw read.

The compact tables are cached to parquet so the expensive normalize+block work
happens once per split.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, build_keys, generate_candidates
from .data_io import build_record_frame

SOURCE_FILES = {1: "source1", 2: "source2", 3: "source3"}


def _read_source(path: str) -> pd.DataFrame:
    """Read one source TSV.

    Everything as string: a postal code like `62704` or a French `bis` must not
    be coerced to a number, and pandas' NA handling would silently blank the
    empty addresses we rely on distinguishing from missing names.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def load_split(
    data_dir: str,
    split: str,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Load and normalize one split (train or test).

    Returns the combined record table -- Source 1 rows first, then Source 2,
    then Source 3 -- plus the row counts needed to slice it.
    """
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"records_{split}.parquet")
        if os.path.exists(cache):
            rec = pd.read_parquet(cache)
            counts = {1: int((rec["src"] == 1).groupby(rec.index).size().iloc[0]) if len(rec) else 0}
            meta_path = os.path.join(cache_dir, f"counts_{split}.json")
            import json
            with open(meta_path) as fh:
                counts = json.load(fh)
            if verbose:
                print(f"  [{split}] loaded cached records {counts}")
            return rec, counts

    frames: List[pd.DataFrame] = []
    counts: Dict[int, int] = {}
    for src in (1, 2, 3):
        path = os.path.join(data_dir, f"{split}_{SOURCE_FILES[src]}.tsv")
        t0 = time.time()
        raw = _read_source(path)
        rec = build_record_frame(raw)
        rec["src"] = np.int8(src)
        del raw
        frames.append(rec)
        counts[src] = len(rec)
        if verbose:
            print(f"  [{split}] source{src}: {len(rec):>9,} rows "
                  f"({time.time() - t0:.1f}s)")

    combined = pd.concat(frames, ignore_index=True, copy=False)
    del frames
    if verbose:
        print(f"  [{split}] total {len(combined):,} normalized records")
    if cache:
        combined.to_parquet(cache, index=False)
        import json
        with open(os.path.join(cache_dir, f"counts_{split}.json"), "w") as fh:
            json.dump(counts, fh)
    return combined, counts


def slice_bounds(counts: Dict[int, int]) -> Dict[str, Tuple[int, int]]:
    """Row ranges for each source inside the combined table."""
    n1, n2, n3 = counts[1], counts[2], counts[3]
    return {
        "s1": (0, n1),
        "cand": (n1, n1 + n2 + n3),
        "s2": (n1, n1 + n2),
        "s3": (n1 + n2, n1 + n2 + n3),
    }


def run_blocking(
    rec: pd.DataFrame,
    counts: Dict[int, int],
    cfg: BlockingConfig,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Blocking keys + candidate generation over a whole split."""
    b = slice_bounds(counts)
    n1, n2, n3 = counts[1], counts[2], counts[3]
    n_cand = n2 + n3

    t0 = time.time()
    s1_mask = np.zeros(len(rec), dtype=bool)
    s1_mask[b["s1"][0]:b["s1"][1]] = True
    keys = build_keys(rec, cfg, s1_row_mask=s1_mask)
    if verbose:
        print(f"  built {len(keys)} blocking key sets ({time.time() - t0:.1f}s)")

    t0 = time.time()
    s1r, candr, stats = generate_candidates(keys, n1, n_cand, cfg)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["n_s1"] = n1
    stats["n_cand"] = n_cand
    stats["full_cross_product"] = n1 * n_cand
    stats["reduction_ratio"] = round(len(s1r) / max(1, n1 * n_cand), 8)
    if verbose:
        print(f"  candidates: {len(s1r):,} pairs "
              f"({stats['reduction_ratio']:.6%} of {n1 * n_cand:,}) "
              f"mean {stats['mean_candidates_per_s1']:.1f}/entity ({time.time() - t0:.1f}s)")
        for k, v in stats["per_key"].items():
            print(f"    {k:14s} pairs={v['pairs']:>10,}  s1_covered={v['s1_covered']:>9,}")
    return s1r, candr, stats
