"""Step 4 -- building the labeled training set.

The honest position
-------------------
There is no ground truth here, so *any* performance number computed on labels
this module invents is optimistic by construction. A model trained on
rule-derived labels and then scored on rule-derived labels mostly learns to
reproduce the rules. That number is not a measure of real-world accuracy.

So this module does two different things and keeps them clearly separate:

1. `build_silver_labels` produces **silver** labels from high-precision rules.
   These are bootstrapping labels. They are used to (a) pre-train, (b) fill the
   unambiguous strata, and (c) sanity-check the model -- never as evidence of
   accuracy.

2. `build_labeling_sheet` produces a **stratified manual sample** for a human to
   label. The strata are chosen so the reviewer's time is spent where the model
   is actually uncertain, not on pairs that are obviously right. This is the
   sample whose labels should replace silver labels for the final model, and the
   sample whose metrics should be quoted.

Feeding real labels back in
---------------------------
`run_pipeline.py --labels reviewed.csv` re-trains on the reviewed file. The
reviewed sheet is the artifact that makes the reported metrics mean something.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Strata, in the order they are reported.
STRATA = [
    "certain_match",     # strong independent agreement
    "probable_match",    # one strong signal, no contradiction
    "ambiguous",         # signals conflict, or all evidence is weak
    "certain_nonmatch",  # hard contradiction
]


def _rule_score(features: pd.DataFrame) -> pd.Series:
    """A transparent, hand-checkable heuristic score in [0, 1].

    Deliberately built from *independent* corroborating signals rather than a
    learned model, so that the silver labels it produces are inspectable: an
    analyst can read which term fired for a given pair and agree or disagree.
    """
    f = features
    s = pd.Series(0.0, index=f.index)

    # --- hard evidence for a match -----------------------------------------
    phone = f["phone10_equal"].fillna(0) + 0.5 * f["phone7_equal"].fillna(0)
    s += 0.45 * phone

    name_strong = f["name_jw_core"].fillna(0)
    s += 0.30 * (name_strong > 0.93).astype(float)
    s += 0.18 * (name_strong > 0.85).astype(float) * (name_strong <= 0.93).astype(float)
    s += 0.20 * f["name_substring"].fillna(0)
    s += 0.15 * f["name_first_tok_equal"].fillna(0)

    geo_ok = f["geo_within_1km"].fillna(0)
    zip_ok = f["zip5_equal"].fillna(0)
    s += 0.20 * (geo_ok + zip_ok).clip(upper=1.0)

    s += 0.10 * f["house_number_equal"].fillna(0)
    s += 0.10 * f["city_equal"].fillna(0)
    s += 0.05 * f["category_equal"].fillna(0)
    s += 0.10 * f["blk_n_keys"] / 7.0

    # --- evidence against a match ------------------------------------------
    # Distance is the single most reliable veto: two businesses 200 km apart
    # with similar names are a franchise, not a duplicate listing.
    dist = f["geo_dist_km"]
    s -= 0.60 * (dist > 50).fillna(False).astype(float)
    s -= 0.25 * (dist > 10).fillna(False).astype(float) * (dist <= 50).fillna(False).astype(float)

    # A confirmed different zip3, or a confirmed different state, is strong
    # evidence of non-match -- but only when both sides actually carry the field.
    s -= 0.35 * (f["zip3_equal"] == 0).fillna(False).astype(float)
    s -= 0.30 * (f["state_equal"] == 0).fillna(False).astype(float)
    s -= 0.30 * (f["city_equal"] == 0).fillna(False).astype(float)
    s -= 0.20 * (f["house_number_equal"] == 0).fillna(False).astype(float)
    s -= 0.25 * (f["name_jw_core"].fillna(1.0) < 0.55).astype(float)

    return s.clip(lower=0.0, upper=1.0)


def build_silver_labels(
    features: pd.DataFrame,
    certain_pos: float = 0.80,
    certain_neg: float = 0.12,
) -> pd.DataFrame:
    """Attach ``stratum`` and ``silver_label`` to the feature frame.

    Only the two extreme strata get a silver label. The middle of the score
    distribution is left unlabeled on purpose: those are the pairs a human needs
    to look at, and inventing a label for them would poison both the model and
    the reported metrics.
    """
    f = features.copy()
    score = _rule_score(f)
    f["rule_score"] = score

    f["stratum"] = np.select(
        [
            score >= certain_pos,
            score <= certain_neg,
        ],
        ["certain_match", "certain_nonmatch"],
        default="probable_match_or_ambiguous",
    )
    # "probable_match" is only claimed when at least one strong signal is
    # present; otherwise the pair is genuinely ambiguous.
    strong = (
        (f["phone10_equal"].fillna(0) > 0)
        | (f["name_jw_core"].fillna(0) > 0.90)
        | (f["zip5_equal"].fillna(0) > 0) & (f["house_number_equal"].fillna(0) > 0)
    )
    f.loc[(f["stratum"] == "probable_match_or_ambiguous") & strong, "stratum"] = "probable_match"
    f.loc[(f["stratum"] == "probable_match_or_ambiguous") & ~strong, "stratum"] = "ambiguous"

    f["silver_label"] = np.select(
        [f["stratum"] == "certain_match", f["stratum"] == "certain_nonmatch"],
        [1, 0],
        default=np.nan,
    )
    return f


def build_labeling_sheet(
    features: pd.DataFrame,
    per_stratum: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample candidate pairs for manual labeling, stratified by difficulty.

    **Sizing the budget.** Default is 1200 pairs. This is a real constraint, not
    a formality: at 220 labels the validation split is ~44 rows and a reported
    precision of 1.00 carries a 95% Wilson interval of roughly [0.84, 1.00] --
    statistically indistinguishable from 0.90. Around 1000-1500 labels with a
    few hundred positives is the point where the interval tightens enough to
    choose a threshold defensibly. If labeling capacity is scarce, cut
    ``certain_match``/``certain_nonmatch`` first: the model learns those regions
    from a handful of examples, and the accuracy comes from the hard strata.

    The allocation is deliberately *not* proportional to stratum size. The
    certain strata only need enough examples to confirm the rules behave; the
    ambiguous stratum is where the decision boundary lives, so it gets the
    largest share. Sampling proportionally would bury the boundary in pairs any
    reviewer would call instantly.

    The output has the required structure -- ``source1_id, candidate_id,
    candidate_source, label`` -- with ``label`` left blank for a human to fill,
    plus the raw field values needed to decide and the features needed to audit
    the rule score.
    """
    if per_stratum is None:
        per_stratum = {
            "certain_match": 150,
            "probable_match": 300,
            "ambiguous": 500,
            "certain_nonmatch": 250,
        }
    rng = np.random.default_rng(seed)

    # Required by the spec and by load_reviewed_labels: never filter these out.
    core_cols = ["s1_id", "candidate_id", "candidate_source", "label"]
    # Everything a reviewer needs to decide, and to audit the rule score.
    optional_cols = [
        "stratum", "rule_score",
        # raw text the reviewer actually reads
        "s1_name_raw", "cand_name_raw",
        "s1_address_raw", "cand_address_raw",
        "s1_phone_raw", "cand_phone_raw",
        "s1_category_raw", "cand_category_raw",
        # the evidence behind the rule score, for auditing
        "name_jw_core", "name_substring", "phone10_equal", "phone7_equal",
        "zip5_equal", "zip3_equal", "city_equal", "state_equal",
        "house_number_equal", "geo_dist_km", "category_equal", "blk_n_keys",
    ]
    keep_cols = core_cols + [c for c in optional_cols if c in features.columns]

    parts = []
    for stratum, n in per_stratum.items():
        pool = features[features["stratum"] == stratum]
        if pool.empty:
            continue
        take = min(n, len(pool))
        idx = rng.choice(pool.index.values, size=take, replace=False)
        parts.append(pool.loc[idx])

    if not parts:
        return pd.DataFrame(columns=keep_cols)

    sheet = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet["label"] = pd.NA  # to be filled by a human: 1 = same entity, 0 = not
    return sheet[keep_cols]


def load_reviewed_labels(
    sheet: pd.DataFrame, reviewed_path: str
) -> pd.DataFrame:
    """Merge human labels back onto the full candidate set.

    Pairs the reviewer left blank stay unlabeled and are excluded from training
    rather than being guessed. Returns the feature frame with a ``label`` column
    populated only where a human decided.
    """
    reviewed = pd.read_csv(reviewed_path, dtype={"candidate_id": str, "s1_id": str})
    reviewed = reviewed[reviewed["label"].notna()].copy()
    reviewed["label"] = reviewed["label"].astype(int)
    if not (set(reviewed["label"].unique()) <= {0, 1}):
        raise ValueError("label column must contain only 0 and 1")

    key = ["s1_id", "candidate_id", "candidate_source"]
    # A duplicated key in the reviewed file would fan out into duplicate feature
    # rows and quietly inflate the training set, so collapse to one label per
    # pair and refuse to guess if the reviewer entered conflicting values.
    dup_mask = reviewed.duplicated(key, keep=False)
    if dup_mask.any():
        conflicting = (
            reviewed[dup_mask].groupby(key)["label"].nunique().gt(1).any()
        )
        if conflicting:
            raise ValueError(
                "reviewed labels contain conflicting values for the same pair; "
                "resolve them before retraining"
            )
        reviewed = reviewed.drop_duplicates(key, keep="first")
    merged = sheet.merge(
        reviewed[key + ["label"]], on=key, how="left", suffixes=("", "_reviewed")
    )
    # `label` from the sheet is all-NA; the reviewed one carries the decision.
    if "label_reviewed" in merged.columns:
        merged["label"] = merged["label_reviewed"]
        merged = merged.drop(columns=["label_reviewed"])
    return merged


def label_distribution(features: pd.DataFrame) -> Dict[str, int]:
    counts = features["stratum"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}
