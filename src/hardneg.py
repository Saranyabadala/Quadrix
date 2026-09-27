"""Hard-negative selection for the match classifier.

Why
---
Random negatives are mostly trivial. A pair whose names share nothing and whose
countries differ is separable by a single feature, so it contributes almost no
gradient: the model satisfies it on the first split and moves on. The negatives
that actually matter sit right next to the positives -- same city, same street,
same brand token, different business -- and those are exactly the pairs a
matcher gets wrong in production.

The baseline sampled negatives uniformly, so the training matrix was dominated
by easy cases while the reported errors were concentrated in the hard residue.

Scope, and why it is deliberately narrow
---------------------------------------
This module **re-weights which negatives enter the training matrix**. It does not
generate new candidate pairs and does not touch the candidate stream, the tune
cache, or the report split. That is a hard constraint, not an oversight:

* the held-out score is computed from the same candidate stream as the baseline,
  so the comparison isolates the training distribution and nothing else;
* the truth map is used only to *discard* a mined negative that turns out to be
  a true match, never to select one. The truth file is exhaustive for the
  sampled entities, so anything absent from it is a genuine non-match -- the
  same premise ``_pair_labels`` already relies on.

Reproducibility
---------------
Every draw goes through an explicitly seeded ``numpy.random.Generator``. There
is no reliance on dict ordering, chunk iteration order, or global random state,
so two runs with the same seed produce byte-identical training matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HardNegConfig:
    """How to build the negative half of the training matrix.

    hard_fraction
        Share of the negative budget drawn from the *hard* band -- the negatives
        the current scorer already rates most similar. These are the pairs just
        below the decision boundary.

    locality_fraction
        Share of the hard band that must additionally agree on country and
        city. Locality agreement without name agreement is the classic
        "unitmate" error: a different business in the same building.

    random_fraction
        Share kept uniformly random. Deliberately non-zero: if every negative
        were hard, the model would see an unrepresentative prior and lose
        calibration on the easy majority that dominates the real candidate
        stream. This term keeps the training distribution anchored.

    max_per_entity
        Cap on hard negatives taken for any one Source 1 entity, so a single
        pathological entity cannot flood the matrix.
    """
    enabled: bool = False
    hard_fraction: float = 0.70
    locality_fraction: float = 0.50
    random_fraction: float = 0.30
    max_per_entity: int = 40
    min_locality_score: float = 0.5


def _composite(frame: pd.DataFrame, score_col: str = "weighted_composite_score") -> np.ndarray:
    if score_col in frame.columns:
        return pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0).to_numpy()
    # No composite available: fall back to an all-equal score so selection
    # degrades to "locality only" rather than to arbitrary.
    return np.zeros(len(frame), dtype=float)


def select_hard_negatives(
    negatives: pd.DataFrame,
    cfg: HardNegConfig,
    budget: int,
    seed: int,
    truth: Optional[dict] = None,
    s1_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Choose `budget` negatives, biased toward the genuinely difficult ones.

    Returns a frame of the same columns as `negatives`. If hard mining is
    disabled, or cannot find enough candidates, this falls back to the uniform
    sample the baseline used, so behaviour is never worse than before.
    """
    n = len(negatives)
    if n == 0 or budget <= 0:
        return negatives.iloc[0:0]
    if not cfg.enabled or budget >= n:
        return negatives.sample(n=min(budget, n), random_state=seed)

    rng = np.random.default_rng(seed)

    # Rank by how similar the pair already looks: the top of this ordering is
    # the hard band.
    scores = _composite(negatives)
    work = negatives.assign(__hn_score=scores)
    work = work.sort_values("__hn_score", ascending=False, kind="mergesort")

    # Locality agreement: same country AND same city, without relying on a
    # precomputed similarity column existing in every code path.
    if {"country_exact", "city_exact"}.issubset(work.columns):
        locality = (
            pd.to_numeric(work["country_exact"], errors="coerce").fillna(0) > 0
        ) & (pd.to_numeric(work["city_exact"], errors="coerce").fillna(0) > 0)
    else:
        locality = pd.Series(False, index=work.index)
    work = work.assign(__hn_local=locality.to_numpy())

    n_hard = int(round(budget * cfg.hard_fraction))
    n_local = int(round(n_hard * cfg.locality_fraction))
    n_plain_hard = max(0, n_hard - n_local)
    n_random = max(0, budget - n_hard)

    picks: list[pd.DataFrame] = []

    local_pool = work[work["__hn_local"]]
    if len(local_pool):
        picks.append(local_pool.head(n_local))

    hard_pool = work[~work["__hn_local"]]
    if len(hard_pool) and n_plain_hard:
        picks.append(hard_pool.head(n_plain_hard))

    if n_random:
        # Sample from what is left after the hard picks, so the random term does
        # not re-draw pairs already selected.
        chosen = pd.concat(picks) if picks else work.iloc[0:0]
        remaining = work.drop(index=chosen.index, errors="ignore")
        if len(remaining):
            take = min(n_random, len(remaining))
            picks.append(remaining.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))

    if not picks:
        return negatives.sample(n=min(budget, n), random_state=seed)

    out = pd.concat(picks, ignore_index=False)

    # Per-entity cap, so no single entity dominates.
    id_col = "s1_entity_id" if "s1_entity_id" in out.columns else None
    if id_col and cfg.max_per_entity > 0:
        out = out.groupby(id_col, group_keys=False).head(cfg.max_per_entity)

    # Truth guard: a "hard negative" that is actually a true match is a label
    # error, and it is the single most damaging mistake this module could make.
    if truth and id_col and "source_record_id" in out.columns:
        def _is_true(row) -> bool:
            cand = str(row["source_record_id"])
            return cand in truth.get(str(row[id_col]), set())
        if len(out):
            out = out[~out.apply(_is_true, axis=1)]

    out = out.drop(columns=[c for c in ("__hn_score", "__hn_local") if c in out.columns])
    if len(out) > budget:
        out = out.sample(n=budget, random_state=seed)
    return out
