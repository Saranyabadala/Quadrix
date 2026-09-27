"""End-to-end contract tests.

The point of these is not to re-test the model -- it is to pin the *shape* of
the deliverable, so a refactor cannot silently change the output schema, drop
the zero-match entities, or collapse the many-to-many structure.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import assemble, evaluate, labels  # noqa: E402


def _fake_scored():
    """A small scored frame covering every case the assembler must handle."""
    return pd.DataFrame({
        "s1_id": ["A", "A", "A", "B", "C", "D"],
        "candidate_id": ["x1", "x2", "y1", "x3", "y2", "x4"],
        "candidate_source": ["source2", "source2", "source3",
                             "source2", "source3", "source2"],
        "score": [0.99, 0.95, 0.9, 0.8, 0.2, 0.05],
        "accepted": [True, True, True, True, False, False],
    })


def test_entity_table_keeps_every_source1_entity_including_unmatched():
    scored = _fake_scored()
    s1 = pd.DataFrame({"business_id": ["A", "B", "C", "D", "E"]})
    table = assemble.build_entity_table(scored, s1, "business_id", threshold=0.5)

    # E has no candidates at all and must still appear.
    assert len(table) == 5
    assert set(table["source1_id"]) == {"A", "B", "C", "D", "E"}
    # C and E have no accepted match -> empty string, not NaN.
    assert table.loc[table.source1_id == "C", "matched_source2_ids"].iloc[0] == ""
    assert table.loc[table.source1_id == "E", "matched_source2_ids"].iloc[0] == ""


def test_entity_table_preserves_many_to_many():
    """An entity matching several records per source must keep all of them."""
    scored = _fake_scored()
    s1 = pd.DataFrame({"business_id": ["A", "B", "C", "D"]})
    table = assemble.build_entity_table(scored, s1, "business_id", threshold=0.5)
    row_a = table[table.source1_id == "A"].iloc[0]
    assert row_a["n_matched_source2"] == 2
    assert set(row_a["matched_source2_ids"].split("|")) == {"x1", "x2"}
    assert row_a["n_matched_source3"] == 1
    assert row_a["n_matched_total"] == 3


def test_entity_table_output_columns_match_spec():
    scored = _fake_scored()
    s1 = pd.DataFrame({"business_id": ["A", "B", "C", "D"]})
    table = assemble.build_entity_table(scored, s1, "business_id", threshold=0.5)
    for col in ("source1_id", "matched_source2_ids", "matched_source3_ids"):
        assert col in table.columns, f"required column {col} missing"


def test_build_pair_scores_applies_threshold():
    scored = assemble.build_pair_scores(_fake_scored().drop(columns=["accepted", "score"]),
                                        scores=[0.99, 0.4, 0.9, 0.1, 0.5, 0.2],
                                        threshold=0.5)
    assert list(scored["accepted"]) == [True, False, True, False, True, False]


def test_dedupe_by_margin_keeps_only_close_scores():
    scored = _fake_scored()
    kept = assemble.dedupe_by_margin(scored, threshold=0.5, margin=0.0)
    # A's two source-2 matches: 0.99 and 0.95. margin 0 keeps only the best.
    assert set(zip(kept.s1_id, kept.candidate_id)) == {("A", "x1"), ("A", "y1"), ("B", "x3")}

    kept_wide = assemble.dedupe_by_margin(scored, threshold=0.5, margin=0.10)
    # margin 0.10 also admits 0.95 (>= 0.99 - 0.10).
    assert ("A", "x2") in set(zip(kept_wide.s1_id, kept_wide.candidate_id))


def test_silver_labels_never_label_the_middle():
    """The ambiguous middle must stay unlabeled -- inventing labels there is
    what poisons the training set."""
    f = pd.DataFrame({
        "phone10_equal": [1.0, 0.0, 0.0],
        "phone7_equal": [1.0, 0.0, 0.0],
        "name_jw_core": [0.98, 0.40, 0.60],
        "name_substring": [1.0, 0.0, 0.0],
        "name_first_tok_equal": [1.0, 0.0, 0.0],
        "geo_within_1km": [1.0, 0.0, 0.0],
        "zip5_equal": [1.0, 0.0, 1.0],
        "zip3_equal": [1.0, 0.0, 1.0],
        "house_number_equal": [1.0, 0.0, 1.0],
        "city_equal": [1.0, 0.0, 1.0],
        "state_equal": [1.0, 0.0, 1.0],
        "category_equal": [1.0, 0.0, 1.0],
        "geo_dist_km": [0.1, 300.0, 0.5],
        "blk_n_keys": [4.0, 0.0, 2.0],
    })
    out = labels.build_silver_labels(f)
    assert out.loc[0, "silver_label"] == 1
    assert out.loc[1, "silver_label"] == 0
    # The middle row agrees on zip/city/house number but is far away and the
    # name disagrees -- it must land in a stratum with no silver label.
    assert pd.isna(out.loc[2, "silver_label"]), "middle must not get a silver label"
    assert out.loc[2, "stratum"] in ("ambiguous", "probable_match")


def test_labeling_sheet_has_required_columns_and_blank_labels():
    f = pd.DataFrame({
        "s1_id": ["A", "B"], "candidate_id": ["x", "y"], "candidate_source": ["source2"] * 2,
        "rule_score": [0.95, 0.05], "name_jw_core": [0.98, 0.3],
        "phone10_equal": [1.0, 0.0], "phone7_equal": [1.0, 0.0],
        "zip5_equal": [1.0, 0.0], "zip3_equal": [1.0, 0.0],
        "city_equal": [1.0, 0.0], "state_equal": [1.0, 0.0],
        "house_number_equal": [1.0, 0.0], "geo_dist_km": [0.1, 500.0],
        "category_equal": [1.0, 0.0], "blk_n_keys": [4.0, 0.0],
        "name_substring": [1.0, 0.0], "name_first_tok_equal": [1.0, 0.0],
        "geo_within_1km": [1.0, 0.0],
        "s1_name_raw": ["Acme", "Acme"], "cand_name_raw": ["Acme LLC", "Acme LLC"],
        "s1_address_raw": ["1 A St", "1 A St"], "cand_address_raw": ["1 A St", "1 A St"],
        "s1_phone_raw": ["555", "555"], "cand_phone_raw": ["555", "555"],
        "s1_category_raw": ["Auto", "Auto"], "cand_category_raw": ["Auto", "Auto"],
    })
    f = labels.build_silver_labels(f)
    sheet = labels.build_labeling_sheet(f, per_stratum={"certain_match": 1, "certain_nonmatch": 1})
    # The four required columns, in the required order at the front.
    assert list(sheet.columns[:4]) == ["s1_id", "candidate_id", "candidate_source", "label"]
    assert sheet["label"].isna().all(), "label must ship blank for the reviewer"


def test_wilson_interval_is_wide_for_small_samples():
    """The whole point of the interval: a tiny sample must not look certain."""
    lo, hi = evaluate.wilson_interval(10, 10)
    assert hi == 1.0
    assert lo < 0.80, f"interval too narrow for 10/10: {lo}"
    lo_big, hi_big = evaluate.wilson_interval(1000, 1000)
    assert lo_big > 0.99 and lo_big > lo, "interval must shrink with more samples"


def test_threshold_recommendation_respects_precision_floor():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 3000
    y = np.concatenate([np.ones(300), np.zeros(n - 300)]).astype(int)
    # Positives score high, negatives low, with overlap.
    s = np.concatenate([
        rng.normal(0.85, 0.10, 300), rng.normal(0.25, 0.15, n - 300)
    ]).clip(0, 1)
    sweep = evaluate.sweep_thresholds(y, s)
    rec = evaluate.recommend_threshold(sweep, target_precision=0.95)
    chosen = rec["precision_target"]
    assert chosen is not None
    assert chosen["precision"] >= 0.95, chosen
    # The chosen point must be interior, not pinned to the very bottom of the grid.
    assert 0.01 < chosen["threshold"] < 0.99, chosen
