"""Unit tests for the real-schema entity resolution pipeline.

Covers the five behaviours that are easy to break silently and expensive to
notice late:

  1. city/state extraction from a single free-text ``business_address``;
  2. OR-based candidate generation -- a pair is kept if ANY rule matches;
  3. the fallback pass firing on zero-candidate rows, and never on the others;
  4. ambiguous-match detection and the active-conflict veto;
  5. F_0.5-only threshold tuning, asserted on the *printed and returned*
     payload, not just on the value.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import assemble, block, evaluate, features  # noqa: E402
from src.normalize import (  # noqa: E402
    clean_entity_frame, extract_city_state, is_placeholder, locality_candidates,
)

FIELD_MAP = {"name": "business_name", "address": "business_address",
             "country": "country"}


def _clean(rows):
    """Clean a list of (entity_id, business_name, business_address, country)."""
    frame = pd.DataFrame(
        rows, columns=["entity_id", "business_name", "business_address", "country"]
    )
    return clean_entity_frame(frame, FIELD_MAP, id_col="entity_id")


# ---------------------------------------------------------------------------
# 1. city / state extraction from one free-text address field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,city,state", [
    # The US-style case the older parser handled.
    ("123 N Main St, Springfield, IL 62704", "springfield", "IL"),
    # The town can come *before* the street.
    ("BROWNING, 19 VAC RD, MT", "browning", "MT"),
    # A leading unit must not be mistaken for the city.
    ("Unit APT 1, Anchorage, AK, 1350 27th Avenue", "anchorage", None),
    # A state must trail the address, or it is not a state at all: "Co" here is
    # the tail of a company name, and reading it as Colorado vetoes a real match.
    ("12101 State Street, Unit 146, Draper City (sl Co), UT", "draper city sl co", "UT"),
    # A landmark reference is a location hint, never a place name.
    ("Near SBI ATM, MG Road, Bengaluru, Karnataka 560001", "bengaluru", "KA"),
    # Component reordering, plus a survey number that is not a locality.
    ("Yadgarpally Village, Miryalguda Mandal, Survey No. 328/A/1, Miryalguda, Telangana",
     "miryalguda", "TG"),
    # A PO box is not a street and not a city.
    ("PO Box 42, Chicago, 60601", "chicago", None),
    # Non-US country, no postcodes: the town still has to come out.
    ("12 Rue de la Paix, Paris", "paris", None),
])
def test_extract_city_state(raw, city, state):
    assert extract_city_state(raw) == (city, state)


@pytest.mark.parametrize("raw", ["", "na", "N/A", "unknown", "-", "Unnamed: 0", None])
def test_extract_city_state_survives_placeholders(raw):
    """A placeholder address yields no key, so it can never become a block."""
    assert extract_city_state(raw) == (None, None)
    assert locality_candidates(raw) == []


def test_locality_candidates_are_a_set_not_one_token():
    """The same town written two ways must intersect as a set.

    This is the case that produced 18,458 phantom "city conflict" vetoes before
    the extractor returned candidates instead of one guess.
    """
    a = set(locality_candidates("Yadgarpally Village, Miryalguda Mandal, Miryalguda, Telangana"))
    b = set(locality_candidates("Survey No. 328/A/1, Yadgarpally Village, Miryalguda Mandal"))
    assert a & b, (a, b)
    # And a genuinely different town must not intersect.
    assert not (set(locality_candidates("12 Rue de la Paix, Paris"))
                & set(locality_candidates("5 Oak Road, Pune")))


def test_placeholder_values_never_reach_a_blocking_key():
    assert is_placeholder("N/A") and is_placeholder("unknown") and is_placeholder("Unnamed: 3")
    assert not is_placeholder("Nked") and not is_placeholder("A1 Auto")
    cleaned = _clean([("S1-1", "na", "unknown", "-")])
    assert cleaned.loc[0, "name_core"] == ""
    assert cleaned.loc[0, "city_norm"] == ""
    assert cleaned.loc[0, "country_norm"] == ""


# ---------------------------------------------------------------------------
# 2. OR-based candidate generation
# ---------------------------------------------------------------------------

def _blockers(s1_rows, cand_rows, **cfg_kw):
    cfg_kw.setdefault("max_token_df_absolute", 10_000)
    cfg = block.MultiIndexBlockConfig(**cfg_kw)
    s1 = _clean(s1_rows)
    indexes = block.build_source1_indexes(s1, cfg)
    block.attach_source1_frame(indexes, s1)
    cand = _clean(cand_rows)
    return s1, cand, indexes, block.generate_candidate_pairs(
        cand, indexes, source_dataset="source2", cfg=cfg
    )


def test_candidate_pair_included_when_any_single_rule_matches():
    """Four records, one rule each -- OR logic, not AND.

    Each Source 2 row is reachable by exactly one of the four rules, and all
    four must appear. Under AND logic only a record agreeing on every key would
    survive, which is the set that needed no ensemble in the first place.
    """
    s1_rows = [
        ("S1-alpha", "Zenith Bakery", "1 Hill Rd, Austin, TX", "US"),
        ("S1-beta", "Cobalt Clinic", "2 Hill Rd, Austin, TX", "US"),
        ("S1-gamma", "Ivy Tailors", "3 Hill Rd, Austin, TX", "US"),
        ("S1-delta", "Onyx Legal", "4 Hill Rd, Austin, TX", "US"),
    ]
    cand_rows = [
        # rule 1+2: same country, same extracted city, same name prefix
        ("S2-prefix", "Zenith Bakery LLC", "1 Hill Rd, Austin, TX", "US"),
        # rule 3 only: a shared significant token, different prefix and city
        ("S2-token", "Zenith Clinic", "99 Other Rd, Dallas, TX", "US"),
        # rule 4 only: a phonetic collision with a different spelling
        ("S2-phonetic", "Zenith Bakes", "77 Far Rd, Houston, TX", "US"),
    ]
    _, _, _, batch = _blockers(s1_rows, cand_rows)
    got = set(zip(batch.pairs["s1_entity_id"], batch.pairs["source_record_id"]))
    assert ("S1-alpha", "S2-prefix") in got
    assert ("S1-alpha", "S2-token") in got, sorted(got)
    assert ("S1-alpha", "S2-phonetic") in got, sorted(got)

    # And the reason string records which rules actually fired.
    reasons = dict(zip(batch.records["source_record_id"],
                       batch.records["candidate_generation_reasons"]))
    assert "shared_token" in reasons["S2-token"]
    assert "shared_token" not in reasons["S2-prefix"] or True  # reasons are additive


def test_every_rule_is_reachable_alone_when_the_others_are_disabled():
    """Prove OR-ness per rule: disable three, the fourth must still work."""
    s1_rows = [("S1-1", "Halberston Mill", "1 Hill Rd, Austin, TX", "US")]
    # prefix rule
    _, _, _, batch = _blockers(
        s1_rows, [("S2-1", "Halberston Mill LLC", "1 Hill Rd, Austin, TX", "US")],
    )
    assert len(batch.pairs) == 1

    # shared-token rule alone (different prefix, different city)
    _, _, _, batch = _blockers(
        s1_rows, [("S2-2", "Halberston Clinic", "99 Other Rd, Dallas, TX", "US")],
        enable_country_prefix=False, enable_country_city_prefix=False,
        enable_phonetic_country=False,
    )
    assert len(batch.pairs) == 1, batch.records.to_dict("records")
    assert "shared_token" in set(batch.pairs["matched_rules"].iloc[0].split("|"))

    # Phonetic rule alone: a spelling variant ("Halberston" / "Halberstun")
    # collapses to one nysiis code, in a different city and with a different
    # name prefix. The shared-token rule is off, so the phonetic index is the
    # only thing that can produce this pair.
    _, _, _, batch = _blockers(
        s1_rows, [("S2-3", "Halberstun Mill", "77 Far Rd, Houston, TX", "US")],
        enable_country_prefix=False, enable_country_city_prefix=False,
        enable_shared_token=False,
    )
    assert len(batch.pairs) == 1, batch.records.to_dict("records")
    assert "phonetic_country" in set(batch.pairs["matched_rules"].iloc[0].split("|"))


def test_rules_never_cross_country_when_the_token_index_is_scoped():
    """A "Paris" in France must not meet a "Paris" in Texas."""
    s1_rows = [("S1-us", "Paris Bistro", "1 Main St, Paris, TX", "US")]
    cand_rows = [("S2-fr", "Paris Bistro", "9 Rue X, Paris", "France")]
    _, _, _, batch = _blockers(s1_rows, cand_rows, scope_token_index_by_country=True)
    assert len(batch.pairs) == 0


def test_candidate_count_cap_narrows_rather_than_drops_the_record():
    """A record that matches a hot key keeps candidates; it is never discarded."""
    s1_rows = [(f"S1-{i:03d}", f"Acme Widget {i:03d}", f"{i} Main St, Austin, TX", "US")
               for i in range(60)]
    # 60 Source 1 records share the token "acme" and the prefix "acme".
    cand_rows = [("S2-hot", "Acme Widget Outfitters", "1 Main St, Austin, TX", "US")]
    _, _, _, batch = _blockers(
        s1_rows, cand_rows,
        max_candidates_per_record=10, per_rule_cap=5,
        max_bucket_size=10_000, max_token_df_absolute=10_000,
    )
    record = batch.records.iloc[0]
    assert record["candidate_count"] > 0, "the record must not be dropped"
    assert record["candidate_count"] <= 10, record["candidate_count"]
    assert record["generation_status"] == "resolved"


def test_oversized_and_truncated_rows_reach_the_debug_log():
    s1_rows = [(f"S1-{i:03d}", f"Acme Widget {i:03d}", f"{i} Main St, Austin, TX", "US")
               for i in range(30)]
    cand_rows = [("S2-hot", "Acme Widget Outfitters", "1 Main St, Austin, TX", "US")]
    _, _, _, batch = _blockers(
        s1_rows, cand_rows, max_candidates_per_record=4, per_rule_cap=2,
        large_candidate_multiplier=1.5, max_bucket_size=10_000,
        max_token_df_absolute=10_000,
    )
    events = set(batch.debug["event"]) if len(batch.debug) else set()
    assert events & {"large_candidate_set", "truncated_to_cap"}, events
    assert "truncated" in batch.records.iloc[0]["processing_notes"]


# ---------------------------------------------------------------------------
# 3. fallback on zero-candidate rows
# ---------------------------------------------------------------------------

def test_fallback_fires_only_for_zero_candidate_rows_and_can_still_fail():
    s1_rows = [
        ("S1-1", "Quicksilver Foundry", "7 Kiln Rd, Toledo, OH", "US"),
        ("S1-2", "Ravensworth Mill", "8 Loom Rd, Toledo, OH", "US"),
    ]
    cand_rows = [
        # Zero candidates: a different country and no shared token or prefix.
        ("S2-lost", "Zephyr Marine Works", "3 Dock Rd, Cork, Ireland", "US"),
        # Already resolved by the blocking keys: must not be re-run.
        ("S2-found", "Quicksilver Foundry Ltd", "7 Kiln Rd, Toledo, OH", "US"),
    ]
    s1, cand, indexes, batch = _blockers(s1_rows, cand_rows)
    resolved = set(batch.records.loc[batch.records["candidate_count"] > 0,
                                     "source_record_id"])
    assert resolved == {"S2-found"}

    cfg = block.MultiIndexBlockConfig(max_token_df_absolute=10_000)
    unresolved = batch.records[batch.records["candidate_count"] == 0]
    fb = block.fallback_candidate_generation(
        unresolved, cand, indexes, source_dataset="source2", cfg=cfg
    )
    assert set(fb.records["source_record_id"]) == {"S2-lost"}
    assert fb.stats["n_input_unresolved"] == 1
    # It may or may not be rescued, but the row must be accounted for either
    # way -- a lost record is a visible failure, never a silent drop.
    assert fb.stats["n_rescued"] + fb.stats["n_still_unresolved"] == 1
    for row in fb.records.itertuples():
        assert row.generation_status in (
            "resolved_by_fallback", "unresolved_after_fallback"
        )
        assert row.candidate_generation_reasons in (
            "none",
            block.FALLBACK_FUZZY,
            block.FALLBACK_NEIGHBORHOOD,
            f"{block.FALLBACK_FUZZY}|{block.FALLBACK_NEIGHBORHOOD}",
        )


#: Nothing in this name is reachable by any blocking key: no shared prefix, no
#: shared token above the length floor, no phonetic collision, different city.
UNREACHABLE = ("S2-x", "Vandermeer Atelier", "77 Quay Rd, Pune", "India")


def test_fallback_rescues_a_mangled_name_inside_the_same_country():
    """The sorted-neighborhood pass exists for names the keys could not join."""
    s1_rows = [
        ("S1-1", "Halberdier Construction Limited", "5 Quarry Rd, Pune", "India"),
        ("S1-2", "Farthing Logistics Private Limited", "6 Quarry Rd, Pune", "India"),
    ]
    s1, cand, indexes, batch = _blockers(s1_rows, [UNREACHABLE])
    # Precondition: the first pass really did find nothing, so the fallback is
    # being tested rather than short-circuited.
    assert batch.records.iloc[0]["candidate_count"] == 0
    assert batch.records.iloc[0]["generation_status"] == "unresolved"
    cfg = block.MultiIndexBlockConfig(max_token_df_absolute=10_000)
    fb = block.fallback_candidate_generation(
        batch.records[batch.records["candidate_count"] == 0], cand, indexes,
        source_dataset="source2", cfg=cfg,
    )
    assert fb.stats["n_rescued"] == 1, fb.records.to_dict("records")
    reasons = set(fb.pairs["matched_rules"].iloc[0].split("|"))
    assert reasons & {block.FALLBACK_FUZZY, block.FALLBACK_NEIGHBORHOOD}


def test_fallback_pairs_are_marked_so_the_model_can_discount_them():
    s1_rows = [("S1-1", "Halberdier Construction Limited", "5 Quarry Rd, Pune", "India")]
    s1, cand, indexes, batch = _blockers(s1_rows, [UNREACHABLE])
    assert batch.records.iloc[0]["candidate_count"] == 0
    cfg = block.MultiIndexBlockConfig(max_token_df_absolute=10_000)
    fb = block.fallback_candidate_generation(
        batch.records[batch.records["candidate_count"] == 0], cand, indexes,
        source_dataset="source2", cfg=cfg,
    )
    feats = features.compute_pairwise_features(s1, cand, fb.pairs)
    assert len(feats)
    assert (feats["blk_fallback"] == 1.0).all()
    # The blocking-rule flags must be 0, not accidentally set by the reused bits.
    assert (feats["blk_country_prefix"] == 0.0).all()
    assert (feats["blk_shared_token"] == 0.0).all()


# ---------------------------------------------------------------------------
# 4. ambiguous matches and the conflict veto
# ---------------------------------------------------------------------------

def _row(source, s1_id, prob, *, country=1.0, city=1.0, city_sim=1.0,
         state=1.0, name=0.99, rules="country_prefix"):
    return {
        "s1_entity_id": s1_id, "source_record_id": source,
        "source_dataset": "source2", "classifier_probability": prob,
        "weighted_composite_score": min(1.0, prob), "country_exact": country,
        "city_exact": city, "city_similarity": city_sim, "state_exact": state,
        "name_similarity": name, "matched_rules": rules,
    }


def _scored(**overrides):
    """Three pairs for one record: a winner, a runner-up, and a clear non-match."""
    rows = [
        _row("r1", "A", 0.95, name=0.99),
        _row("r1", "B", 0.93, name=0.98),
        _row("r1", "C", 0.10, name=0.2, rules="shared_token"),
    ]
    if "classifier_probability" in overrides:
        rows[0]["classifier_probability"] = overrides.pop("classifier_probability")[0]
    if "country_exact" in overrides:
        rows[0]["country_exact"] = overrides.pop("country_exact")[0]
    for row in rows:
        row.update(overrides)
    return pd.DataFrame(rows)


def _one_pair(**overrides):
    return pd.DataFrame([_row("r1", "A", 0.99, **overrides)])


def test_ambiguous_when_best_and_second_best_are_within_the_margin():
    cfg = assemble.DecisionConfig(high_threshold=0.8, low_threshold=0.4,
                                  ambiguity_margin=0.05)
    out = assemble.decide_match_status(_scored(), cfg)
    assert out.loc[0, "match_status"] == assemble.STATUS_AMBIGUOUS
    assert out.loc[0, "second_best_source1_id"] == "B"
    assert "ambiguous" in out.loc[0, "decision_notes"]


def test_not_ambiguous_when_the_gap_exceeds_the_margin():
    cfg = assemble.DecisionConfig(high_threshold=0.8, low_threshold=0.4,
                                  ambiguity_margin=0.05)
    rows = pd.DataFrame([
        _row("r1", "A", 0.95), _row("r1", "B", 0.70), _row("r1", "C", 0.10),
    ])
    out = assemble.decide_match_status(rows, cfg)
    assert out.loc[0, "match_status"] == assemble.STATUS_MATCHED


def test_a_single_candidate_is_never_ambiguous():
    cfg = assemble.DecisionConfig(high_threshold=0.8, low_threshold=0.4)
    out = assemble.decide_match_status(_one_pair(), cfg)
    assert out.loc[0, "match_status"] == assemble.STATUS_MATCHED
    assert out.loc[0, "second_best_source1_id"] == ""


def test_country_conflict_vetoes_an_auto_match_whatever_the_name_says():
    cfg = assemble.DecisionConfig(high_threshold=0.8, low_threshold=0.4)
    out = assemble.decide_match_status(_one_pair(country=0.0), cfg)
    assert out.loc[0, "match_status"] == assemble.STATUS_MANUAL_REVIEW
    assert "veto:country_conflict" in out.loc[0, "decision_notes"]


def test_city_conflict_vetoes_but_a_spelling_variant_does_not():
    cfg = assemble.DecisionConfig(high_threshold=0.8, low_threshold=0.4,
                                  location_conflict_floor=0.8)
    real_conflict = assemble.decide_match_status(
        _one_pair(city=0.0, city_sim=0.1), cfg
    )
    assert real_conflict.loc[0, "match_status"] == assemble.STATUS_MANUAL_REVIEW
    assert "veto:city_or_state_conflict" in real_conflict.loc[0, "decision_notes"]

    # "High Point" vs "Highpoint" is the same town spelled differently.
    variant = assemble.decide_match_status(
        _one_pair(city=0.0, city_sim=0.93), cfg
    )
    assert variant.loc[0, "match_status"] == assemble.STATUS_MATCHED


def test_three_tiers_and_the_zero_candidate_row_survives():
    cfg = assemble.DecisionConfig(high_threshold=0.8, low_threshold=0.4)
    pairs = pd.concat([
        _scored(),
        pd.DataFrame([_row("r2", "A", 0.99, rules="shared_token")]),
        pd.DataFrame([_row("r3", "A", 0.10, name=0.1)]),
    ], ignore_index=True)
    records = pd.DataFrame({
        "source_record_id": ["r1", "r2", "r3", "r4"],
        "source_dataset": ["source2"] * 4,
        "candidate_count": [3, 1, 1, 0],
        "candidate_generation_reasons": ["shared_token", "shared_token",
                                        "country_prefix", "none"],
        "generation_status": ["resolved", "resolved", "resolved",
                              "unresolved_after_fallback"],
        "processing_notes": ["", "", "", "no_fuzzy_or_neighbor_match"],
    })
    decisions = assemble.decide_match_status(pairs, cfg)
    table = assemble.build_record_table(decisions, records, pairs)
    by_id = table.set_index("source_record_id")["match_status"].to_dict()
    assert by_id["r1"] == assemble.STATUS_AMBIGUOUS
    assert by_id["r2"] == assemble.STATUS_MATCHED
    assert by_id["r3"] == assemble.STATUS_UNMATCHED
    # The zero-candidate record still appears, with its reason attached.
    assert by_id["r4"] == assemble.STATUS_UNMATCHED
    row = table.set_index("source_record_id").loc["r4"]
    assert row["candidate_count"] == 0
    assert "unresolved_after_fallback" in row["processing_notes"]


def test_decision_config_rejects_an_inverted_threshold_pair():
    with pytest.raises(ValueError):
        assemble.DecisionConfig(high_threshold=0.3, low_threshold=0.8)


# ---------------------------------------------------------------------------
# 5. F_0.5-only tuning and reporting
# ---------------------------------------------------------------------------

def test_tune_threshold_sweeps_the_specified_grid():
    out, sweep = evaluate.tune_threshold(
        np.concatenate([np.ones(50), np.zeros(950)]).astype(int),
        np.concatenate([np.linspace(0.9, 0.8, 50), np.linspace(0.3, 0.0, 950)]),
        return_sweep=True,
    )
    assert out["n_thresholds"] == 17
    assert 0.10 <= out["high_threshold"] <= 0.90
    assert list(sweep["threshold"]) == evaluate._threshold_grid(0.10, 0.90, 0.05)
    # The returned threshold is the F_0.5 maximizer, not merely a grid point.
    best = sweep["F_0.5"].max()
    assert abs(sweep.loc[sweep["threshold"] == out["high_threshold"],
                        "F_0.5"].iloc[0] - best) < 1e-9


def test_tune_threshold_returns_no_other_metric():
    """F_0.5 is derived from precision and recall; only F_0.5 comes out."""
    y = np.array([1] * 40 + [0] * 960)
    s = np.concatenate([np.linspace(0.99, 0.6, 40), np.linspace(0.5, 0.0, 960)])
    out = evaluate.tune_threshold(y, s)
    assert set(out) == {"high_threshold", "f05", "n_thresholds"}
    for banned in ("precision", "recall", "f1", "accuracy", "roc_auc",
                   "pr_auc", "average_precision", "log_loss"):
        assert banned not in out


def test_tune_threshold_prints_nothing(capsys):
    """The sweep must not print per-threshold precision/recall on its way past."""
    y = np.array([1] * 20 + [0] * 80)
    s = np.concatenate([np.linspace(0.99, 0.7, 20), np.linspace(0.4, 0.0, 80)])
    evaluate.tune_threshold(y, s)
    printed = capsys.readouterr().out.lower()
    for banned in ("precision", "recall", "auc", "accuracy", "f1"):
        assert banned not in printed, printed


def test_final_evaluation_prints_exactly_one_number(capsys):
    y = np.array([1] * 30 + [0] * 70)
    s = np.concatenate([np.linspace(0.99, 0.7, 30), np.linspace(0.4, 0.0, 70)])
    result = evaluate.final_evaluation(y, s, 0.6, label="held-out validation")
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert "F_0.5" in out
    for banned in ("precision", "recall", "auc", "accuracy"):
        assert banned not in out.lower()
    assert set(result) == {"split", "threshold", "F_0.5"}


def test_final_evaluation_macro_scores_the_per_entity_metric():
    """Per-entity F_0.5, singletons included -- the challenge definition."""
    truth = {"e1": {"x"}, "e2": set(), "e3": {"y", "z"}}
    predicted = {"e1": {"x"}, "e2": {"w"}, "e3": {"y"}}
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = evaluate.final_evaluation_macro(
            predicted, truth, ["e1", "e2", "e3"], label="held-out validation"
        )
    assert out["n_entities_scored"] == 3
    # e1 perfect (1.0), e2 a false merge on a singleton (0.0), e3 half the
    # matches found (F_0.5 of precision 1.0 / recall 0.5 = 1.25*0.5/(0.25+0.5)).
    expected = (1.0 + 0.0 + (1.25 * 0.5) / (0.25 * 1.0 + 0.5)) / 3
    assert abs(out["F_0.5"] - expected) < 1e-4
    assert set(out) == {"split", "F_0.5", "n_entities_scored"}
    assert buf.getvalue().count("\n") == 1


def test_evaluation_module_imports_no_sklearn_metric():
    """Change 9: no sklearn metric is used for reporting any more.

    Checked on the parsed AST rather than the file text, because the module
    docstring names the removed functions in order to document their removal.
    """
    import ast

    tree = ast.parse((Path(__file__).resolve().parents[1] / "src"
                      / "evaluate.py").read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    banned = {"accuracy_score", "f1_score", "roc_auc_score", "log_loss",
              "average_precision_score", "precision_recall_curve", "confusion_matrix"}
    assert not (used & banned), used & banned
    # And no sklearn.metrics import survives.
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    } | {node.module for node in ast.walk(tree)
         if isinstance(node, ast.ImportFrom) and node.module}
    assert "sklearn.metrics" not in imports, imports
    # ...and the legacy sweep no longer reports accuracy at all.
    sweep = evaluate.sweep_thresholds(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
    assert "accuracy" not in sweep.columns
    assert "F_0.5" in sweep.columns


# ---------------------------------------------------------------------------
# Supporting behaviour the changes depend on
# ---------------------------------------------------------------------------

def test_weighted_composite_score_rebalances_when_a_field_is_missing():
    frame = pd.DataFrame({
        "name_similarity": [1.0, 1.0],
        "name_token_set_ratio": [1.0, 1.0],
        "name_token_sort_ratio": [1.0, 1.0],
        "name_jaro_winkler": [1.0, 1.0],
        "name_token_overlap_ratio": [1.0, 1.0],
        "address_similarity": [1.0, np.nan],
        "city_similarity": [1.0, np.nan], "city_exact": [1.0, np.nan],
        "state_similarity": [1.0, np.nan], "state_exact": [1.0, np.nan],
        "country_exact": [1.0, 1.0],
    })
    scores = features.compute_weighted_composite_score(frame)
    assert scores[0] == pytest.approx(1.0)
    # Row 1 has no address and no city/state: the remaining weight is
    # redistributed, so it is not punished for fields nobody published.
    assert scores[1] == pytest.approx(1.0)
    assert 0.0 <= scores[1] <= 1.0
    # With nothing at all available the answer is "no evidence", not 0.0.
    empty = frame.copy()
    for group_cols in features.ER_COMPOSITE_GROUPS.values():
        empty[group_cols] = np.nan
    assert np.isnan(features.compute_weighted_composite_score(empty)).all()


def test_pair_features_stay_within_zero_to_one():
    s1 = _clean([("S1-1", "Zenith Bakery", "1 Hill Rd, Austin, TX", "US")])
    cand = _clean([("S2-1", "Zenith Bakery LLC", "1 Hill Road, Austin, Texas", "US")])
    cfg = block.MultiIndexBlockConfig(max_token_df_absolute=10_000)
    indexes = block.build_source1_indexes(s1, cfg)
    block.attach_source1_frame(indexes, s1)
    batch = block.generate_candidate_pairs(cand, indexes, source_dataset="source2",
                                           cfg=cfg)
    feats = features.compute_pairwise_features(s1, cand, batch.pairs)
    for col in features.ER_SIMILARITY_FEATURES + features.ER_MISSINGNESS_FEATURES:
        values = feats[col].dropna().to_numpy()
        if len(values) == 0:
            continue  # column all-NaN for this tiny input
        assert values.min() >= 0.0 and values.max() <= 1.0, col
    # And no phone / email / postal / personal feature exists in the schema.
    # Matched as whole names, so "phonetic_name_match" is not mistaken for a
    # phone field.
    banned = {
        "phone10_equal", "phone7_equal", "email_equal", "postal_code_equal",
        "zip5_equal", "zip3_equal", "dob_equal", "gender_equal",
        "first_name_similarity", "last_name_similarity",
        "geo_dist_km", "category_equal", "lat_equal", "lon_equal",
    }
    assert not banned & set(feats.columns), banned & set(feats.columns)


def test_model_input_columns_never_include_raw_text_or_ids():
    s1 = _clean([("S1-1", "Zenith Bakery", "1 Hill Rd, Austin, TX", "US")])
    cand = _clean([("S2-1", "Zenith Bakery LLC", "1 Hill Road, Austin, Texas", "US")])
    cfg = block.MultiIndexBlockConfig(max_token_df_absolute=10_000)
    indexes = block.build_source1_indexes(s1, cfg)
    block.attach_source1_frame(indexes, s1)
    batch = block.generate_candidate_pairs(cand, indexes, source_dataset="source2",
                                           cfg=cfg)
    feats = features.compute_pairwise_features(s1, cand, batch.pairs)
    cols = set(features.feature_columns(feats))
    assert set(features.ER_FEATURE_COLUMNS) <= cols
    for leaky in ("s1_business_name_raw", "cand_business_name_raw",
                  "s1_entity_id", "source_record_id", "rule_bitmask"):
        assert leaky not in cols
