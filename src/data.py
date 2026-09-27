"""Configuration and source loading.

`load_sources` is the only place that touches the raw CSVs. It applies the
column mapping from config/schema.yaml, runs the normalizer, and returns cleaned
frames that carry both the normalized columns and the original text (the latter
purely for the manual-labeling sheet -- raw text is never a model feature).

The `*_entity_source` helpers at the bottom are the equivalent entry points for
the minimal real schema (entity_id / business_name / business_address /
country) and the only ones that stream: Source 2 and Source 3 are read with
`read_csv(chunksize=...)` so a 5M-row source never has to exist in memory at
once.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Sequence, Set

import pandas as pd
import yaml

from .normalize import clean_entity_frame, clean_frame


def load_config(path: str = "config/schema.yaml") -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_sources(
    config: dict,
    root: str = ".",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Read and clean every source declared in the config.

    Returns ``{source_name: cleaned_frame}``. Each cleaned frame has the
    normalized columns from ``normalize.clean_frame`` plus ``<field>_raw``
    display columns and the source's own id column.
    """
    cleaned: Dict[str, pd.DataFrame] = {}

    for name, spec in config["sources"].items():
        raw_path = os.path.join(root, spec["path"])
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"missing input for {name}: {raw_path}\n"
                f"Generate a benchmark dataset first:  python run_pipeline.py --generate"
            )

        # Keep every column as string first: a zip like 62704 or a phone like
        # 5550142 would otherwise be parsed as an int and lose its leading
        # zeros, and the normalizer cannot recover a lost zero.
        df = pd.read_csv(raw_path, dtype=str, keep_default_na=False, na_values=[])
        id_col = spec["id_col"]
        if id_col not in df.columns:
            raise KeyError(f"{name}: id_col {id_col!r} not in {list(df.columns)[:10]}...")

        field_map = spec["fields"]
        norm = clean_frame(df, field_map)

        # Carry the raw text through for human review, keyed by canonical name.
        for canonical, col in field_map.items():
            if col and col in df.columns:
                norm[f"{canonical}_raw"] = df[col].fillna("")
            else:
                norm[f"{canonical}_raw"] = ""

        norm[id_col] = df[id_col].values
        # A stable positional key: every downstream index refers to this.
        norm = norm.reset_index(drop=True)
        norm.insert(0, "_pos", range(len(norm)))

        if verbose:
            miss = norm["n_missing"].mean()
            print(
                f"  {name:8s} {len(norm):5d} rows | "
                f"mean missing fields/entity: {miss:.2f} | "
                f"geo available: {int(norm['lat'].notna().sum())}"
            )
        cleaned[name] = norm

    return cleaned


def load_truth(path: str = "data/_truth/truth_pairs.csv") -> Optional[pd.DataFrame]:
    """Load the hidden truth file if it exists.

    Only the synthetic benchmark writes one. On real data this returns None and
    every metric that needs ground truth is reported as unavailable rather than
    quietly substituting a proxy.
    """
    if not os.path.exists(path):
        return None
    truth = pd.read_csv(path, dtype=str)
    truth["label"] = 1
    return truth


# ---------------------------------------------------------------------------
# Real schema (entity_id / business_name / business_address / country)
# ---------------------------------------------------------------------------

def load_entity_source(
    path: str,
    spec: Dict[str, object],
    root: str = ".",
    sep: str = "\t",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read and normalize one real-schema source file in full.

    Only Source 1 is ever loaded this way: it is the smaller of the three
    (2.2M rows) and it has to be resident anyway, because every candidate probe
    hits its indexes. Source 2/3 go through `iter_entity_chunks`.

    ``nrows`` is a development shortcut (subsample a source to iterate
    quickly); it is never used on a scoring run.
    """
    full_path = os.path.join(root, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"missing input file: {full_path}")
    id_col = str(spec.get("id_col") or "entity_id")
    fields = dict(spec.get("fields") or {})
    raw = pd.read_csv(
        full_path, sep=sep, dtype=str, keep_default_na=False, na_values=[],
        nrows=nrows,
    )
    if id_col not in raw.columns:
        raise KeyError(
            f"{path}: id_col {id_col!r} not in {list(raw.columns)[:10]}"
        )
    cleaned = clean_entity_frame(raw, fields, id_col=id_col)
    return cleaned.reset_index(drop=True)


def iter_entity_chunks(
    path: str,
    spec: Dict[str, object],
    root: str = ".",
    sep: str = "\t",
    chunksize: int = 50_000,
    nrows: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Stream a real-schema source file chunk by chunk.

    ``pandas.read_csv(chunksize=...)`` is what keeps Source 2 and Source 3
    affordable: each chunk is normalized, probed against the Source 1 indexes,
    scored and written, then released. Peak memory is the Source 1 index plus
    one chunk plus that chunk's candidate pairs -- not 10M rows of joined text.

    ``nrows`` is honoured across chunks (not per chunk) so a development
    subsample really is N rows in total.
    """
    full_path = os.path.join(root, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"missing input file: {full_path}")
    id_col = str(spec.get("id_col") or "entity_id")
    fields = dict(spec.get("fields") or {})
    remaining = nrows
    reader = pd.read_csv(
        full_path, sep=sep, dtype=str, keep_default_na=False, na_values=[],
        chunksize=chunksize,
    )
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                return
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        cleaned = clean_entity_frame(chunk, fields, id_col=id_col)
        # Positional key within the chunk: the pair table indexes features by it.
        cleaned = cleaned.reset_index(drop=True)
        yield cleaned
        if remaining is not None:
            remaining -= len(chunk)


def load_ground_truth_map(path: str, root: str = ".") -> Dict[str, Set[str]]:
    """``{source1_entity_id: {matched ids}}`` from the challenge truth file.

    Reuses `metrics.load_ground_truth`, which also preserves the full entity
    list order so the singleton rows -- the ones with an empty match list --
    survive. Dropping them would silently change the denominator of the macro
    average and inflate the score.
    """
    from .metrics import load_ground_truth

    return load_ground_truth(os.path.join(root, path))[0]

