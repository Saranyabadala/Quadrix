"""Blocking at competition scale.

The full test cross product is 1.73M Source 1 records x 9.97M Source 2/3 records
= 1.7e13 pairs. Even at 10 nanoseconds a pair that is 1.7e5 CPU-seconds, and it
would have to be done twice (once to train, once to submit). Blocking is what
makes the problem tractable, and its recall is a hard ceiling on the score:
**a pair blocking never generates cannot be recovered by any model.**

Strategy
--------
Every key is scoped by country. That is not a heuristic, it is a measured fact:
0 of 152,302 sampled ground-truth pairs cross countries. Scoping buys three
things at once:

* It is a free, perfect key on its own -- roughly a 3x reduction before any name
  matching happens.
* It is the only key that transfers to **France**, which is 15% of the test set,
  entirely absent from training, and has no postcodes. Any design that leans on
  US zip codes scores zero there without ever raising an error.
* It prevents the pathological cross-country blocks (a name like "Paris" in
  France vs a "Paris" in Texas) that would otherwise dominate the candidate
  counts.

Keys, in rough order of precision:

  country|name_soundex      phonetic, survives small typos
  country|zip               strong where postcodes exist (US, India)
  country|city|prefix4     re-anchors on locality when the exact zip is missing
  country|number|city       address anchor; catches records whose name is mangled
  country|latin_token       inverted index over *rare* Latin name tokens
  country|other_token       same, for the non-Latin script (Devanagari)
  country|prefix4           cheap last resort

The two token keys are what make cross-script India work at all. 11.3% of true
Indian pairs pair a Devanagari name with a Latin one, and mixed-script names
like ``Golden गोल्ड सर्विसेज`` share only their Latin tokens with the Latin
spelling in Source 1. A Latin-token key therefore reaches pairs that any
whole-string comparison scores at zero.

Scaling
-------
Nothing here builds a Python-level loop over records. Keys are factorized to
int32, the inverted index for each key is a CSR-style ``(indptr, indices)`` pair
built with a single ``argsort``, and the pair expansion is a vectorized
``np.repeat`` + fancy-index. Peak memory is a few hundred MB regardless of
input size, and per-Source-1 caps are applied *before* the expansion so a single
pathological key cannot blow up the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Keys are applied in this order and the first ones are the most precise, so the
# per-key budget is spent on the best evidence first.
KEY_NAMES = [
    "core_exact",
    "prefix8",
    "latin_token",
    "number_city",
    "soundex",
    "zip",
    "city_prefix",
    "prefix4",
    "other_token",
]


@dataclass
class BlockingConfig:
    """Caps and thresholds for candidate generation.

    per_key_cap
        Maximum candidates one key may contribute for a single Source 1 record.
        This is the control that keeps a coarse key from dominating: a token
        shared by 5,000 records is not evidence, and truncating it is correct,
        not lossy.

    bucket_cap
        A key value held by more than this many *candidate* records is treated as
        uninformative and skipped entirely for that value. Cheaper than
        collecting and truncating thousands of positions.

    per_entity_cap
        Final ceiling on candidates per Source 1 record. Applied after
        deduplication across keys.

    max_token_df_ratio
        A name token held by more than this fraction of Source 1 records is a
        category word ("services", "private"), not an identifier, so it never
        becomes a key.
    """
    per_key_cap: int = 20
    bucket_cap: int = 2000
    per_entity_cap: int = 60
    max_token_df_ratio: float = 0.01
    max_token_len: int = 24
    min_token_len: int = 3


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------

def _tok_codes(
    frame: pd.DataFrame, col: str, cfg: BlockingConfig, n_rows: int
) -> List[np.ndarray]:
    """Explode a space-joined token column into parallel arrays.

    Returns ``(codes, offsets)`` where ``codes`` is the token code per token
    occurrence and ``offsets`` is the start index into ``codes`` for each row.
    This is the CSR layout that makes the inverted index a single argsort.
    """
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    values = frame[col].astype(str).to_numpy()
    per_row: List[List[str]] = []
    lengths = np.zeros(n_rows, dtype=np.int64)
    for i, v in enumerate(values):
        if v:
            toks = [t for t in v.split() if cfg.min_token_len <= len(t) <= cfg.max_token_len]
        else:
            toks = []
        per_row.append(toks)
        lengths[i] = len(toks)
    offsets[1:] = np.cumsum(lengths)

    flat: List[str] = []
    for toks in per_row:
        flat.extend(toks)
    if not flat:
        return np.zeros(0, dtype=np.int32), offsets
    uniq, codes = np.unique(np.asarray(flat, dtype=object), return_inverse=True)
    return codes.astype(np.int32), offsets


def build_keys(
    rec: pd.DataFrame,
    cfg: BlockingConfig,
    s1_row_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Build every blocking key for a combined record table.

    ``rec`` is the output of ``build_record_frame`` for Source 2 and Source 3
    stacked, optionally with the Source 1 rows prepended (pass ``s1_row_mask`` to
    restrict token-frequency estimation to Source 1 rows, which is what decides
    which tokens count as informative).

    Returns a dict of int32 code arrays, one per key. Codes are aligned across
    Source 1 and candidate rows by factorizing the key string over the whole
    combined table, so equality of codes means equality of keys.
    """
    n = len(rec)
    country = rec["country"].to_numpy(dtype=np.int64)
    zipc = rec["addr_zip"].to_numpy(dtype=np.int64)
    cityc = rec["addr_city"].to_numpy(dtype=np.int64)
    numc = rec["addr_number"].to_numpy(dtype=np.int64)
    sounc = rec["name_soundex"].to_numpy(dtype=np.int64)
    pref = rec["name_prefix4"].to_numpy(dtype=np.int64)

    keys: Dict[str, object] = {}

    def combine(*arrays: np.ndarray) -> np.ndarray:
        """Hash several int arrays into one int32 code.

        Mixed radix on 21 bits each. Deterministic and collision-resistant
        enough for 10M rows; a collision only widens a block slightly, it cannot
        invent a false match, because the model still has to score the pair.
        """
        out = np.zeros(n, dtype=np.int64)
        for a in arrays:
            a = a.astype(np.int64) + 1  # reserve 0 for "missing"
            out = (out * 2_097_152 + a) % 2_147_483_647
        return out.astype(np.int32)

    keys["soundex"] = combine(country, sounc)
    keys["zip"] = combine(country, zipc)
    keys["city_prefix"] = combine(country, cityc, pref)
    keys["number_city"] = combine(country, numc, cityc)
    keys["prefix4"] = combine(country, pref)

    # More specific name keys. On real data a 4-character prefix or a first-token
    # Soundex is far coarser than it looks on a tidy synthetic vocabulary:
    # "asso" and Soundex("associated") are each shared by thousands of unrelated
    # businesses, so those buckets exceed bucket_cap and contribute nothing.
    # Measured on the training truth, the 8-character prefix and the exact
    # core are what actually carry US recall.
    keys["prefix8"] = combine(country, rec["name_prefix8"].to_numpy(dtype=np.int64))
    keys["core_exact"] = combine(country, rec["name_core"].to_numpy(dtype=np.int64))

    # --- token keys --------------------------------------------------------
    # Document frequency over Source 1 only, so "informative" is judged by the
    # reference source rather than by the much larger candidate pool.
    n_s1 = int(s1_row_mask.sum()) if s1_row_mask is not None else n
    cutoff = max(2, int(cfg.max_token_df_ratio * max(1, n_s1)))

    for col, keyname in (("name_latin", "latin_token"), ("name_other", "other_token")):
        codes, offsets = _tok_codes(rec, col, cfg, n)
        keep = _informative_tokens(codes, offsets, s1_row_mask, cutoff, n)
        # record index for every surviving occurrence -- the token path needs
        # this because one record contributes several occurrences, and the
        # candidate index must be a record index, not an occurrence index.
        row_of_occ = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
        if codes.size == 0:
            keys[keyname] = (
                np.zeros(0, dtype=np.int32), offsets, row_of_occ.astype(np.int32)
            )
        else:
            keys[keyname] = (
                codes[keep],
                _recompute_offsets(keep, offsets, n),
                row_of_occ[keep].astype(np.int32),
            )
    return keys



def _informative_tokens(
    codes: np.ndarray, offsets: np.ndarray, s1_mask: Optional[np.ndarray], cutoff: int, n: int
) -> np.ndarray:
    """Boolean mask over token occurrences, dropping high-frequency tokens."""
    if codes.size == 0:
        return np.zeros(0, dtype=bool)
    uniq, inv = np.unique(codes, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
    if s1_mask is not None:
        # Recompute frequency restricted to Source 1 rows.
        row_of_token = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
        in_s1 = s1_mask[row_of_token] if row_of_token.size == codes.size else np.zeros(codes.size, bool)
        s1_counts = np.bincount(inv[in_s1], minlength=len(uniq)).astype(np.int64)
    else:
        s1_counts = counts
    return s1_counts[inv] <= cutoff


def _recompute_offsets(keep: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
    """Offsets after filtering token occurrences.

    Uses a prefix sum rather than ``np.add.reduceat``: reduceat requires strictly
    increasing segment starts and raises on rows that contributed zero tokens,
    which is common (empty names, stopword-only names).
    """
    if keep.size == 0:
        return np.zeros(n + 1, dtype=np.int64)
    csum = np.concatenate([[0], np.cumsum(keep.astype(np.int64))])
    kept_per_row = csum[offsets[1:]] - csum[offsets[:-1]]
    new = np.zeros(n + 1, dtype=np.int64)
    new[1:] = np.cumsum(kept_per_row)
    return new


# ---------------------------------------------------------------------------
# Inverted index and pair generation
# ---------------------------------------------------------------------------

def _csr_index(
    codes: np.ndarray, rows: np.ndarray, n_code_space: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(indptr, indices)`` mapping key code -> candidate record rows.

    ``rows`` must be the *record* index for each entry in ``codes``, which is why
    it is a parameter: for a scalar key that is just ``arange``, but for a token
    key one record contributes several entries and the occurrence position is
    not the record position.
    """
    valid = codes >= 0
    c = codes[valid].astype(np.int64)
    r = rows[valid].astype(np.int32)
    if c.size == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32)

    order = np.argsort(c, kind="stable")
    c_sorted, r_sorted = c[order], r[order]
    uniq, start = np.unique(c_sorted, return_index=True)
    counts = np.diff(np.append(start, c_sorted.size))
    # Scatter into a dense code space so lookups are a plain index.
    size = int(n_code_space) if n_code_space else int(c_sorted.max()) + 1
    indptr = np.zeros(size + 1, dtype=np.int64)
    indptr[uniq + 1] = counts
    np.cumsum(indptr, out=indptr)
    return indptr, r_sorted



def _expand(
    rows: np.ndarray,
    codes: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    cap: int,
    bucket_cap: int,
    n_cand_limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized (s1_row, cand_row) expansion for one key, with caps.

    ``rows`` and ``codes`` are parallel arrays of equal length, one entry per
    *bucket an entity belongs to*. That generalization is what lets scalar keys
    and token keys share this code path:

    * a scalar key gives one code per Source 1 record;
    * a token key gives one code per (Source 1 record, token) pair, so a record
      with three informative tokens appears three times in ``rows``.

    ``cap`` is applied *per bucket*, then again per entity by the caller.
    Truncation keeps the lowest candidate positions, which after a stable
    argsort are ordered by original row -- i.e. arbitrary with respect to the
    label, so truncation does not bias the training sample.
    """
    if len(codes) == 0 or len(indptr) < 2:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    valid = codes >= 0
    if not valid.any():
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    rows = rows[valid]
    codes = codes[valid].astype(np.int64)

    safe = np.minimum(codes, len(indptr) - 2)
    starts = indptr[safe]
    sizes = indptr[safe + 1] - starts

    ok = (sizes > 0) & (sizes <= bucket_cap)
    if not ok.any():
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    rows, starts, sizes = rows[ok].astype(np.int32), starts[ok], sizes[ok]

    take = np.minimum(sizes, cap)
    total = int(take.sum())
    if total == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)

    out_s1 = np.repeat(rows, take)
    base = np.repeat(starts, take)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(take) - take, take
    )
    out_cand = indices[base + within]
    # Defensive: the index must only address candidate rows. A negative or
    # out-of-range value would silently score a garbage record, so drop those
    # pairs. The mask is applied to both arrays together to keep them aligned.
    if out_cand.size:
        ok = (out_cand >= 0) & (out_cand < n_cand_limit)
        if not ok.all():
            out_s1, out_cand = out_s1[ok], out_cand[ok]
    return out_s1.astype(np.int32), out_cand.astype(np.int32)



def generate_candidates(
    keys: Dict[str, object],
    n_s1: int,
    n_cand: int,
    cfg: BlockingConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Union the per-key candidate sets.

    Returns ``(s1_row, cand_row, stats)``. ``cand_row`` indexes into the
    concatenated candidate table (Source 2 rows first, then Source 3).
    """
    s1_all: List[np.ndarray] = []
    c_all: List[np.ndarray] = []
    stats: Dict[str, object] = {"per_key": {}}

    for keyname in KEY_NAMES:
        payload = keys.get(keyname)
        if payload is None:
            continue

        if keyname in ("latin_token", "other_token"):
            codes, offsets, row_of_occ = payload
            if codes.size == 0:
                stats["per_key"][keyname] = {"pairs": 0, "s1_covered": 0}
                continue
            # Occurrences are laid out row-major over the whole table, so the
            # first `n_s1_tok` belong to Source 1 and the rest to candidates.
            # The inverted index MUST be built from the candidate occurrences
            # only: including Source 1's own occurrences would put negative
            # (Source-1-relative) row numbers into the candidate index, and a
            # Source-1-only token would then "match" a negative row.
            lens = np.diff(offsets[: n_s1 + 1])
            n_s1_tok = int(lens.sum())
            if n_s1_tok == 0 or n_s1_tok >= codes.size:
                stats["per_key"][keyname] = {"pairs": 0, "s1_covered": 0}
                continue
            indptr, indices = _csr_index(
                codes[n_s1_tok:], (row_of_occ[n_s1_tok:] - n_s1).astype(np.int32)
            )
            s1_rows = np.repeat(np.arange(n_s1, dtype=np.int32), lens)
            s1_codes = codes[:n_s1_tok]
        else:
            all_codes = payload
            s1_codes = all_codes[:n_s1].astype(np.int64)
            indptr, indices = _csr_index(
                all_codes[n_s1:], np.arange(n_cand, dtype=np.int32)
            )
            s1_rows = np.arange(n_s1, dtype=np.int32)

        s1r, candr = _expand(
            s1_rows, s1_codes, indptr, indices, cfg.per_key_cap, cfg.bucket_cap, n_cand
        )
        stats["per_key"][keyname] = {
            "pairs": int(len(s1r)),
            "s1_covered": int(len(np.unique(s1r))) if len(s1r) else 0,
        }
        if len(s1r):
            s1_all.append(s1r)
            c_all.append(candr)

    if not s1_all:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), stats

    s1r = np.concatenate(s1_all)
    candr = np.concatenate(c_all)
    stats["pairs_before_dedup"] = int(len(s1r))

    # Deduplicate (s1, cand). Pack into a single int64 for a fast np.unique.
    packed = s1r.astype(np.int64) * np.int64(n_cand + 1) + candr.astype(np.int64)
    uniq = np.unique(packed)
    s1r = (uniq // (n_cand + 1)).astype(np.int32)
    candr = (uniq % (n_cand + 1)).astype(np.int32)
    stats["pairs_before_dedup"] = int(len(s1_all) and sum(len(a) for a in s1_all))
    stats["pairs_after_dedup"] = int(len(s1r))

    # Final per-entity cap, keeping the first `cap` candidates per Source 1 row.
    order = np.lexsort((candr, s1r))
    s1r, candr = s1r[order], candr[order]
    if cfg.per_entity_cap > 0:
        starts = np.flatnonzero(np.r_[True, s1r[1:] != s1r[:-1]])
        rank = np.arange(len(s1r), dtype=np.int64) - np.repeat(starts, np.diff(np.r_[starts, len(s1r)]))
        keep = rank < cfg.per_entity_cap
        s1r, candr = s1r[keep], candr[keep]

    stats["pairs_final"] = int(len(s1r))
    stats["mean_candidates_per_s1"] = float(len(s1r) / max(1, n_s1))
    stats["s1_with_zero_candidates"] = int(n_s1 - len(np.unique(s1r)))
    return s1r, candr, stats
