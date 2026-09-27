"""Tests for hard-negative selection.

The invariants that matter are not "does it select hard pairs" but "can it
corrupt a label". A hard-negative miner that accidentally includes a true match
is worse than no miner at all: it trains the model to reject a real match, and
the damage is invisible in the training matrix and only shows up as lost recall.
So the tests below are weighted toward the safety properties.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.hardneg import HardNegConfig, select_hard_negatives  # noqa: E402


def _neg(n=200, seed=0):
    """Negatives where composite score correlates with 'hardness'."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "s1_entity_id": [f"S1-{i % 20}" for i in range(n)],
        "source_record_id": [f"S2-{i:05d}" for i in range(n)],
        "weighted_composite_score": rng.random(n),
        "country_exact": np.ones(n),
        "city_exact": np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)]),
        "name_similarity": rng.random(n),
    })


def test_disabled_falls_back_to_uniform_sample():
    n = _neg()
    out = select_hard_negatives(n, HardNegConfig(enabled=False), budget=50, seed=1)
    assert len(out) == 50
    # Must be a plain sample, not the top-50 by score.
    top = n.nlargest(50, "weighted_composite_score")["source_record_id"].tolist()
    assert set(out["source_record_id"]) != set(top)


def test_enabled_prefers_high_composite_scores():
    n = _neg()
    out = select_hard_negatives(
        n, HardNegConfig(enabled=True, hard_fraction=0.9, locality_fraction=0.0),
        budget=40, seed=1,
    )
    assert len(out) == 40
    assert out["weighted_composite_score"].mean() > n["weighted_composite_score"].mean(), (
        "hard mining must skew toward the high-score band"
    )


def test_never_selects_a_true_match():
    """The critical safety property."""
    n = _neg()
    top = n.nlargest(1, "weighted_composite_score").iloc[0]
    victim, owner = top["source_record_id"], top["s1_entity_id"]
    # The truth guard is keyed by Source 1 entity, so declare the match under the
    # entity that actually owns the row.
    truth = {owner: {victim}}
    out = select_hard_negatives(
        n, HardNegConfig(enabled=True), budget=40, seed=1, truth=truth
    )
    assert victim not in set(out["source_record_id"]), "a true match leaked into negatives"


def test_truth_guard_is_scoped_to_its_own_entity():
    """A match declared under a *different* entity must not be dropped.

    Guards against over-deletion: the lookup is per (s1_entity, candidate), so
    an unrelated entity's truth cannot silently shrink the negative pool.
    """
    n = _neg()
    victim = n.nlargest(1, "weighted_composite_score")["source_record_id"].iloc[0]
    out_wrong = select_hard_negatives(
        n, HardNegConfig(enabled=True), budget=40, seed=1,
        truth={"S1-does-not-own-this": {victim}},
    )
    out_none = select_hard_negatives(
        n, HardNegConfig(enabled=True), budget=40, seed=1, truth=None
    )
    assert len(out_wrong) == len(out_none), "unrelated truth entry changed the selection"


def test_reproducible_for_a_fixed_seed():
    n = _neg()
    cfg = HardNegConfig(enabled=True)
    a = select_hard_negatives(n, cfg, budget=60, seed=7)
    b = select_hard_negatives(n, cfg, budget=60, seed=7)
    assert list(a["source_record_id"]) == list(b["source_record_id"]), "seeded run not deterministic"


def test_respects_per_entity_cap():
    n = _neg(n=400)
    out = select_hard_negatives(
        n, HardNegConfig(enabled=True, max_per_entity=3), budget=200, seed=1
    )
    counts = out.groupby("s1_entity_id").size()
    assert counts.max() <= 3, f"per-entity cap breached: {counts.max()}"


def test_never_exceeds_budget():
    n = _neg(n=120)
    out = select_hard_negatives(n, HardNegConfig(enabled=True), budget=30, seed=1)
    assert len(out) <= 30


def test_empty_and_degenerate_inputs():
    empty = _neg(0)
    assert len(select_hard_negatives(empty, HardNegConfig(enabled=True), budget=10, seed=1)) == 0
    n = _neg(10)
    # budget larger than the pool: return everything, do not crash
    assert len(select_hard_negatives(n, HardNegConfig(enabled=True), budget=999, seed=1)) == 10


def test_missing_composite_column_degrades_gracefully():
    n = _neg(50).drop(columns=["weighted_composite_score"])
    out = select_hard_negatives(n, HardNegConfig(enabled=True), budget=20, seed=1)
    assert len(out) > 0, "must still select something when the score column is absent"
