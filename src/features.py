"""Step 3 -- pairwise feature engineering.

For every candidate pair we compute a row of features that the classifier turns
into a match probability. Three principles drove the design:

1. **Compare components, not just blobs.** A single fuzzy score over a
   concatenated name+address string hides which part agreed. Decomposed features
   (house number, street name, city, zip, unit) let the model learn that a
   matching house number in the same zip is strong evidence even when the street
   name was misspelled, and that a *unit* mismatch is a downgrade rather than a
   disqualifier.

2. **Feed the model the reason a feature is unavailable.** A NaN in
   ``name_jaro_winkler`` because the name is missing on one side is different
   from a NaN because the names are simply dissimilar. Tree models route NaN
   natively, so the *indicator* columns alongside each feature are what let it
   distinguish "no evidence" from "negative evidence".

3. **Let multiple metrics disagree.** Jaro-Winkler rewards a shared prefix,
   Levenshtein penalizes length change, TF-IDF cosine is corpus-aware (so a rare
   brand word counts for more than "shop"), and token overlap is immune to word
   order. For names like "Acme Bicycle" vs "Acme Bike" these disagree sharply,
   and the disagreement is itself signal.

Missing values are written as ``np.nan`` and never imputed: the gradient
boosters used here split on NaN natively, and imputation would invent values for
the very pairs we know least about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rapidfuzz.distance import JaroWinkler as _rf_jw
    from rapidfuzz.distance import Levenshtein as _rf_lev

    def jaro_winkler(a: str, b: str) -> float:
        if not a or not b:
            return float("nan")
        return float(_rf_jw.similarity(a, b))

    def levenshtein_ratio(a: str, b: str) -> float:
        """1 - normalized edit distance, in [0, 1]."""
        if not a or not b:
            return float("nan")
        dist = _rf_lev.distance(a, b)
        longest = max(len(a), len(b))
        return 1.0 - dist / float(longest)

except Exception:  # pragma: no cover - pure-python fallbacks
    def jaro_winkler(a: str, b: str) -> float:
        if not a or not b:
            return float("nan")
        return _jaro_winkler_py(a, b)

    def levenshtein_ratio(a: str, b: str) -> float:
        if not a or not b:
            return float("nan")
        m, n = len(a), len(b)
        if abs(m - n) > max(m, n) * 0.5:
            return 0.0
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            prev = cur
        return 1.0 - prev[n] / float(max(m, n))

    def _jaro_winkler_py(s1: str, s2: str, p: float = 0.1) -> float:
        len1, len2 = len(s1), len(s2)
        if not len1 or not len2:
            return 0.0
        window = max(len1, len2) // 2 - 1
        if window < 0:
            window = 0
        s1_flags = [False] * len1
        s2_flags = [False] * len2
        matches = 0
        for i in range(len1):
            lo = max(0, i - window)
            hi = min(i + window + 1, len2)
            for j in range(lo, hi):
                if not s2_flags[j] and s1[i] == s2[j]:
                    s1_flags[i] = s2_flags[j] = True
                    matches += 1
                    break
        if matches == 0:
            return 0.0
        transpositions = 0
        k = 0
        for i in range(len1):
            if s1_flags[i]:
                while not s2_flags[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
        transpositions //= 2
        jaro = (matches / len1 + matches / len2
                + (matches - transpositions) / matches) / 3.0
        prefix = 0
        for a, b in zip(s1, s2):
            if a != b:
                break
            prefix += 1
            if prefix == 4:
                break
        return jaro + prefix * p * (1 - jaro)


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km. NaN if either coordinate is missing."""
    try:
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (lat1, lon1, lat2, lon2)):
            return float("nan")
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = p2 - p1
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))
    except (TypeError, ValueError):
        return float("nan")


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """NaN when either side has no tokens -- 'no evidence' is not 'zero overlap'."""
    if not a or not b:
        return float("nan")
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / float(len(union))


def _containment(a: Sequence[str], b: Sequence[str]) -> float:
    """Overlap relative to the *smaller* token set.

    "Acme Bike Shop" vs "Acme Bicycle" has low Jaccard (2 of 4 union tokens) but
    full containment of the shorter set, which is the right reading when one
    record simply carries more words.
    """
    if not a or not b:
        return float("nan")
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return float("nan")
    return len(sa & sb) / float(min(len(sa), len(sb)))


def _exact(a, b) -> float:
    """1.0 if both present and equal, 0.0 if both present and different, NaN if
    either is missing."""
    if a is None or b is None or a == "" or b == "":
        return float("nan")
    return 1.0 if a == b else 0.0


def build_tfidf(all_names: List[str]):
    """Fit one TF-IDF vectorizer over the union of all three sources' names.

    Fitting on the union (not per source) is what makes the cosine meaningful:
    if Source 3's vocabulary were fit separately, a word common in that corpus
    would not be down-weighted relative to the rest of the world.

    Character n-grams are used alongside word n-grams because business names are
    short and heavily abbreviated, and "1234 North Main" vs "1234 N Main St" is
    a character-level problem more than a word-level one.
    """
    corpus = [n for n in all_names if n]
    if not corpus:
        return None, None
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
        min_df=1, lowercase=True, token_pattern=r"(?u)\b\w+\b",
    )
    try:
        word_vec.fit(corpus)
    except ValueError:
        return None, None
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, min_df=1,
    )
    try:
        char_vec.fit(corpus)
    except ValueError:
        char_vec = None
    return (word_vec, char_vec), word_vec, char_vec


def compute_features(
    s1: pd.DataFrame,
    cand_by_source: Dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    s1_id_col: str,
    cand_id_cols: Dict[str, str],
) -> pd.DataFrame:
    """Build the full feature matrix, one row per candidate pair.

    Returns a frame with the identifying columns, the engineered features, and
    the raw display fields needed for the manual-labeling sheet.
    """
    if candidates.empty:
        return pd.DataFrame()

    # ---- TF-IDF over the union of all names --------------------------------
    # Rows are stacked in the same positional order the candidate indices refer
    # to: Source 1 first, then each candidate source in turn.
    sources = list(cand_by_source.keys())
    name_docs = list(s1["name_norm"].fillna(""))
    name_docs += list(s1["name_core"].fillna(""))
    for src in sources:
        name_docs += list(cand_by_source[src]["name_norm"].fillna(""))
        name_docs += list(cand_by_source[src]["name_core"].fillna(""))
    vecs, word_vec, char_vec = build_tfidf(name_docs)

    n1 = len(s1)
    offsets = {}
    running = n1 * 2
    for src in sources:
        offsets[src] = running
        running += len(cand_by_source[src]) * 2

    if word_vec is not None:
        all_names = (list(s1["name_norm"]) + list(s1["name_core"])
                     + [n for src in sources for n in cand_by_source[src]["name_norm"]]
                     + [n for src in sources for n in cand_by_source[src]["name_core"]])
        all_names = [n if isinstance(n, str) and n else "" for n in all_names]
        word_mat = word_vec.transform(all_names)
        # L2-normalize so a dot product *is* the cosine similarity.
        from sklearn.preprocessing import normalize
        word_mat = normalize(word_mat)
        char_mat = None
        if char_vec is not None:
            char_mat = normalize(char_vec.transform(all_names))
    else:
        word_mat = char_mat = None

    rows: List[Dict] = []
    cache: Dict[Tuple, float] = {}

    def tfidf_cos(mat, i, j) -> float:
        if mat is None:
            return float("nan")
        return float(mat[i].multiply(mat[j]).sum())

    for rec in candidates.itertuples():
        sp = int(rec.s1_pos)
        src = rec.candidate_source
        cp = int(rec.cand_pos)
        off = offsets[src]
        a = s1.iloc[sp]
        b = cand_by_source[src].iloc[cp]

        feat: Dict[str, object] = {
            "s1_id": a[s1_id_col],
            "candidate_id": b[cand_id_cols[src]],
            "candidate_source": src,
        }

        # ---- name similarity ------------------------------------------------
        a_norm, b_norm = a["name_norm"], b["name_norm"]
        a_core, b_core = a["name_core"], b["name_core"]

        feat["name_jw_norm"] = jaro_winkler(a_norm, b_norm)
        feat["name_jw_core"] = jaro_winkler(a_core, b_core)
        feat["name_lev_norm"] = levenshtein_ratio(a_norm, b_norm)
        feat["name_lev_core"] = levenshtein_ratio(a_core, b_core)

        # Word n-grams over the core form, char n-grams over the full form.
        feat["name_tfidf_cos"] = (
            tfidf_cos(word_mat, sp, off + cp) if word_mat is not None else float("nan")
        )
        feat["name_tfidf_char_cos"] = (
            tfidf_cos(char_mat, n1 + sp, n1 + off + cp) if char_mat is not None else float("nan")
        )

        a_toks, b_toks = a["name_tokens"], b["name_tokens"]
        feat["name_tok_jaccard"] = _jaccard(a_toks, b_toks)
        feat["name_tok_containment"] = _containment(a_toks, b_toks)
        feat["name_tok_count_diff"] = (
            float(abs(len(a_toks) - len(b_toks))) if a_toks and b_toks else float("nan")
        )
        feat["name_len_ratio"] = (
            float(min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm)))
            if a_norm and b_norm else float("nan")
        )
        # First token is the brand, which is the most stable part of a name.
        feat["name_first_tok_equal"] = _exact(
            a_norm.split()[0] if a_norm else None, b_norm.split()[0] if b_norm else None
        )
        from .block import soundex
        feat["name_soundex_equal"] = _exact(
            soundex(a_core) if a_core else None, soundex(b_core) if b_core else None
        )
        # Does either name contain the other outright? Catches "Acme" vs
        # "Acme Bicycle" where fuzzy metrics under-score the shorter string.
        # bool() is required: `x in y and a and b` evaluates to a *string* when
        # truthy, which float() cannot convert.
        feat["name_substring"] = float(
            bool(a_core) and bool(b_core) and (a_core in b_core or b_core in a_core)
        )

        # ---- address similarity --------------------------------------------
        feat["addr_jw"] = jaro_winkler(a["address_norm"] or "", b["address_norm"] or "")
        feat["addr_lev"] = levenshtein_ratio(a["address_norm"] or "", b["address_norm"] or "")
        feat["house_number_equal"] = _exact(a["house_number"], b["house_number"])
        feat["street_name_jw"] = jaro_winkler(a["street_name"] or "", b["street_name"] or "")
        feat["street_name_equal"] = _exact(a["street_name"], b["street_name"])
        feat["street_suffix_equal"] = _exact(a["street_suffix"], b["street_suffix"])
        feat["street_dir_equal"] = _exact(a["street_dir"], b["street_dir"])
        feat["city_equal"] = _exact(a["city_norm"], b["city_norm"])
        feat["state_equal"] = _exact(a["state_norm"], b["state_norm"])
        feat["zip5_equal"] = _exact(a["zip5"], b["zip5"])
        feat["zip3_equal"] = _exact(a["zip3"], b["zip3"])
        # PO box records share no street; comparing the box number is the only
        # meaningful address comparison available.
        feat["pobox_equal"] = _exact(a["po_box"], b["po_box"])

        # Unit is a *downgrade* signal, not a mismatch: same building, different
        # suite usually means the same business or a branch of it. A boolean
        # both-have-and-differ lets the model learn that as a mild penalty
        # rather than treating it like a different address.
        both_units = bool(a["unit"]) and bool(b["unit"])
        feat["unit_conflict"] = float(both_units and a["unit"] != b["unit"]) if both_units else float("nan")
        feat["unit_both_present"] = float(both_units) if both_units else float("nan")

        # ---- geography ------------------------------------------------------
        dist = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        feat["geo_dist_km"] = dist
        # log1p compresses the tail: 5 km vs 25 km matters, 500 km vs 520 km does
        # not, and a linear feature would be dominated by the latter.
        feat["geo_log_dist"] = math.log1p(dist) if not math.isnan(dist) else float("nan")
        feat["geo_within_1km"] = float(dist <= 1.0) if not math.isnan(dist) else float("nan")
        feat["geo_within_10km"] = float(dist <= 10.0) if not math.isnan(dist) else float("nan")

        # ---- phone ----------------------------------------------------------
        feat["phone10_equal"] = _exact(a["phone10"], b["phone10"])
        feat["phone7_equal"] = _exact(a["phone7"], b["phone7"])

        # ---- category -------------------------------------------------------
        feat["category_equal"] = _exact(a["category_norm"], b["category_norm"])

        # ---- missingness ----------------------------------------------------
        # Per-field "either side missing" flags, plus a count. The model needs
        # these because the same similarity value means different things when
        # one side is blank: phone equality between two missing numbers is
        # undefined, not a match.
        for fld in ("name", "address", "city", "state", "zip", "phone", "category"):
            ma = bool(a[f"{fld}_missing"])
            mb = bool(b[f"{fld}_missing"])
            feat[f"{fld}_missing_either"] = float(ma or mb)
            feat[f"{fld}_missing_both"] = float(ma and mb)
        feat["n_missing_s1"] = float(a["n_missing"])
        feat["n_missing_cand"] = float(b["n_missing"])
        feat["n_missing_total"] = float(a["n_missing"]) + float(b["n_missing"])

        # ---- blocking provenance -------------------------------------------
        # Which keys produced the pair. A pair that agrees on phone *and* zip
        # *and* name is corroborated by independent evidence; a pair that only
        # shares a loose name token is not.
        for key in ("phone10", "zip5", "zip3_name", "soundex", "name_prefix", "geo_cell", "token"):
            feat[f"blk_{key}"] = float(bool(getattr(rec, key)))
        feat["blk_n_keys"] = float(
            sum(bool(getattr(rec, k)) for k in
                ("phone10", "zip5", "zip3_name", "soundex", "name_prefix", "geo_cell", "token"))
        )

        # ---- display fields for the labeling sheet --------------------------
        feat["s1_name_raw"] = a.get("name_raw", "")
        feat["s1_address_raw"] = a.get("address_raw", "")
        feat["s1_phone_raw"] = a.get("phone_raw", "")
        feat["s1_category_raw"] = a.get("category_raw", "")
        feat["cand_name_raw"] = b.get("name_raw", "")
        feat["cand_address_raw"] = b.get("address_raw", "")
        feat["cand_phone_raw"] = b.get("phone_raw", "")
        feat["cand_category_raw"] = b.get("category_raw", "")

        rows.append(feat)

    return pd.DataFrame(rows)


# Columns that must never reach the classifier: identifiers, raw text, or the
# blocking-provenance labels are kept separate so `FEATURE_COLUMNS` can be
# derived mechanically and a leak cannot slip in by accident.
NON_FEATURE_COLUMNS = {
    "s1_id", "candidate_id", "candidate_source", "label", "stratum",
    "s1_name_raw", "s1_address_raw", "s1_phone_raw", "s1_category_raw",
    "cand_name_raw", "cand_address_raw", "cand_phone_raw", "cand_category_raw",
    "truth_label", "s1_pos", "cand_pos",
    # Real-schema (entity_id / business_name / business_address / country)
    # identifiers, positions, raw text and provenance. None of these may be
    # features: a raw business_name column would let the model memorise
    # training rows instead of comparing them.
    "s1_entity_id", "source_record_id", "source_dataset", "s1_row", "cand_pos",
    "rule_bitmask", "matched_rules", "s1_business_name_raw", "s1_business_address_raw",
    "s1_country_raw", "cand_business_name_raw", "cand_business_address_raw",
    "cand_country_raw", "pair_reasons", "n_rules_matched_int",
}


def feature_columns(features: pd.DataFrame) -> List[str]:
    """Numeric columns safe to feed the model, in a stable order."""
    cols = []
    for c in features.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(features[c]):
            cols.append(c)
    return sorted(cols)


# ===========================================================================
# Real-schema pair features (entity_id / business_name / business_address /
# country only)
#
# The feature set above was designed for a schema with phone, category, zip and
# coordinates. None of those exist in the real dataset, so this section is the
# feature set that actually matches the data, and `compute_features` above is
# left in place for the richer benchmark.
#
# Everything is derived from the four available columns: the name, the address
# blob, the country, and the city/state recovered from the address blob. There
# is deliberately no phone/email/postal/first-last-name/DOB/gender feature --
# inventing a column that the data does not contain produces a constant 0.0,
# which looks like a feature to the model and teaches it nothing.
# ===========================================================================

#: The Change 5 feature list, in one place so the model matrix and the docs
#: cannot drift apart. Every value is already on 0-1, or NaN for "no evidence".
ER_SIMILARITY_FEATURES: List[str] = [
    # name
    "name_similarity",              # mean of the three name metrics below
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_jaro_winkler",
    "name_token_overlap_ratio",     # |A n B| / min(|A|, |B|)
    # address
    "address_similarity",           # token_set_ratio over the whole address
    # extracted city / state
    "city_similarity", "city_exact",
    "state_similarity", "state_exact",
    # country
    "country_exact",
    # phonetic agreement
    "phonetic_name_match",
    # character-level (typo / abbreviation / word-order robustness)
    "name_char_jaro_winkler", "name_levenshtein_ratio", "name_len_ratio",
    "name_token_count_ratio", "address_char_jaro_winkler",
    # domain mined from the name
    "domain_exact", "domain_suffix_match", "domain_present_either",
    # script-aware, for the cross-script Indian pairs
    "name_latin_token_jaccard", "name_cross_script",
]

ER_MISSINGNESS_FEATURES: List[str] = [
    "name_missing_either", "name_placeholder_either",
    "address_missing_either", "address_parse_failed_either",
    "city_missing_either", "state_missing_either", "country_missing_either",
    "n_fields_missing_total",
]

#: Blocking provenance, one column per rule plus a count. Kept as features
#: because a pair corroborated by three independent keys is not the same
#: evidence as one that shares a single loose token.
ER_BLOCKING_RULES: List[str] = [
    "country_prefix", "country_city_prefix", "shared_token",
    "phonetic_country", "exact_entity_id",
]

ER_PROVENANCE_FEATURES: List[str] = (
    ["n_rules_matched"] + [f"blk_{r}" for r in ER_BLOCKING_RULES] + ["blk_fallback"]
)

ER_FEATURE_COLUMNS: List[str] = (
    ER_SIMILARITY_FEATURES + ER_MISSINGNESS_FEATURES + ER_PROVENANCE_FEATURES
)

#: Group -> the features that feed the explainable weighted score (Change 6).
ER_COMPOSITE_GROUPS: Dict[str, List[str]] = {
    "name": [
        "name_similarity", "name_token_set_ratio", "name_token_sort_ratio",
        "name_jaro_winkler", "name_token_overlap_ratio",
    ],
    "address": ["address_similarity"],
    "city_state": [
        "city_similarity", "city_exact", "state_similarity", "state_exact",
    ],
    "country": ["country_exact"],
}


def _safe(arr: np.ndarray) -> np.ndarray:
    """object array -> array of python strings, with None/NaN as ''."""
    out = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        out[i] = "" if v is None or (isinstance(v, float) and v != v) else str(v)
    return out


def _safe_bool(arr: np.ndarray) -> np.ndarray:
    """Any column -> a bool array, with None/NaN/'' as False."""
    out = np.zeros(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if v is None:
            continue
        if isinstance(v, float) and v != v:
            continue
        out[i] = bool(v)
    return out


def _pairwise_ratio(
    a: np.ndarray, b: np.ndarray, scorer, scale: float = 1.0
) -> np.ndarray:
    """Element-wise fuzzy similarity for two equal-length string arrays.

    Uses rapidfuzz's `cpdist`, which is C-implemented and multi-threaded, so a
    10M-pair chunk is a vectorised op rather than a 10M-iteration Python loop.
    (`cdist` is the *cartesian* variant and would allocate n x n -- exactly the
    full cross product this pipeline must never build.)
    """
    if len(a) == 0:
        return np.zeros(0, dtype=np.float64)
    from rapidfuzz import process

    if len(a) != len(b):  # pragma: no cover - defensive
        raise ValueError(f"pairwise inputs must align: {len(a)} vs {len(b)}")
    return np.asarray(
        process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32),
        dtype=np.float64,
    ) * scale


def _masked(values: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Blank out a similarity where either side has no value.

    'No evidence' must not read as 'zero similarity' -- a missing address is a
    different fact from two different addresses, and the model has a
    missingness flag to tell them apart. 0.0 is still used for a genuine
    mismatch, which is what keeps the mean over available columns meaningful.
    """
    present = (a != "") & (b != "")
    out = np.where(present, values, np.nan)
    return out


def _nanmean(mat: np.ndarray, axis: int = 1) -> np.ndarray:
    """Mean ignoring NaN, NaN where the whole slice is NaN, and no warning.

    `np.nanmean` emits a RuntimeWarning for an all-NaN slice and that is the
    *normal* case here: "both sides had no address" is expected data, not an
    error. Counting valid entries explicitly also makes the NaN case explicit
    rather than emergent.
    """
    valid = ~np.isnan(mat)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, mat, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / np.maximum(counts, 1)
    return np.where(counts > 0, mean, np.nan)


def _token_overlap(a_toks: List, b_toks: List) -> float:
    """Overlap relative to the smaller token set (containment), NaN if empty.

    Containment rather than Jaccard: "Acme Bike Shop" vs "Acme Bicycle" has
    low Jaccard but full containment of the shorter set, which is the right
    reading when one record simply carries more words.
    """
    sa, sb = set(a_toks or ()), set(b_toks or ())
    if not sa or not sb:
        return float("nan")
    return len(sa & sb) / float(min(len(sa), len(sb)))


def _memoize(fn, values: np.ndarray) -> np.ndarray:
    """Apply a per-string function once per *distinct* string.

    Phonetic codes and name metrics are recomputed per pair otherwise, but the
    number of distinct business names in a chunk is orders of magnitude smaller
    than the number of pairs, so a dict memo turns the cost into a hash lookup.
    """
    cache: Dict[str, object] = {}
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        key = values[i]
        if key in cache:
            out[i] = cache[key]
        else:
            out[i] = cache[key] = fn(key)
    return out


def _has_non_latin(arr: np.ndarray) -> np.ndarray:
    """True where the string contains a non-Latin letter.

    Used to flag cross-script name pairs, which need different evidence
    than same-script ones.
    """
    return np.array([
        any("\u0370" <= ch <= "\u1cff" or "\u0900" <= ch <= "\u0dff"
            or "\u4e00" <= ch <= "\u9fff" for ch in str(x))
        for x in arr
    ], dtype=bool)


def _positions(pairs: pd.DataFrame, col: str) -> np.ndarray:
    """A pair's positional indices as int64, whatever dtype it arrived as.

    Concatenating two `CandidateBatch` frames promotes an all-empty int32
    column to object, and indexing with an object array raises rather than
    misbehaving -- but the same cast also absorbs a float column produced by a
    row that went missing, which would otherwise index with NaN and raise a much
    more confusing error three lines later.
    """
    if col not in pairs.columns:
        return np.zeros(len(pairs), dtype=np.int64)
    values = pd.to_numeric(pairs[col], errors="coerce").fillna(-1)
    return values.to_numpy(dtype=np.int64)


def _locality_sets(values: np.ndarray) -> List[frozenset]:
    """One frozenset of locality candidates per row, memoized by the candidate.

    `frozenset` and `isdisjoint` are C-level, so the per-pair set comparison in
    `compute_pairwise_features` costs a dict lookup plus one call. That matters:
    a naive Python set construction per pair would dominate the whole feature
    stage at tens of millions of pairs.
    """
    cache: Dict[tuple, frozenset] = {}
    out: List[frozenset] = []
    for value in values:
        key = tuple(value or ())
        cached = cache.get(key)
        if cached is None:
            cached = cache[key] = frozenset(t for t in key if t)
        out.append(cached)
    return out


def compute_pairwise_features(
    s1: pd.DataFrame,
    chunk: pd.DataFrame,
    pairs: pd.DataFrame,
    s1_id_col: str = "entity_id",
    cand_id_col: str = "entity_id",
    include_raw: bool = True,
) -> pd.DataFrame:
    """One feature row per candidate pair, using only the four real columns.

    Parameters
    ----------
    s1
        Cleaned Source 1 frame (see `normalize.clean_entity_frame`).
    chunk
        Cleaned Source 2/3 frame the pairs were generated from.
    pairs
        Candidate pairs with ``s1_row`` (positional row in ``s1``),
        ``cand_pos`` (positional row in ``chunk``) and the rule bitmask. This
        is the frame `block.generate_candidate_pairs` returns, so the positional
        indices are the contract between Stage 2 and Stage 3.

    Returns a frame containing ``ER_FEATURE_COLUMNS`` plus identifiers and (when
    ``include_raw``) the original text, which the review and output files need
    but the model must never see -- both are listed in `NON_FEATURE_COLUMNS`.
    """
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    if pairs is None or len(pairs) == 0:
        return pd.DataFrame()

    s1_rows = _positions(pairs, "s1_row")
    cand_rows = _positions(pairs, "cand_pos")
    # Defensive: a pair pointing outside its frame would silently score a
    # different record, which is the worst failure mode in this stage.
    s1_rows = np.clip(s1_rows, 0, max(0, len(s1) - 1))
    cand_rows = np.clip(cand_rows, 0, max(0, len(chunk) - 1))

    def col(frame: pd.DataFrame, name: str) -> np.ndarray:
        if name in frame.columns:
            return frame[name].to_numpy()
        return np.full(len(frame), "", dtype=object)

    def boolcol(frame: pd.DataFrame, name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        return _safe_bool(frame[name].to_numpy())

    def pick(frame: pd.DataFrame, name: str, idx: np.ndarray) -> np.ndarray:
        """Gather `name` at positional `idx`, always returning len(idx) rows.

        `col` alone is not safe for this: for a missing column it returns an
        array sized to the *frame*, so indexing it with per-pair positions walks
        off the end (or, worse, silently reads the wrong row when the frame is
        shorter than the pair count). Clamping inside the gather keeps the
        per-pair contract that every feature below relies on.
        """
        if name in frame.columns:
            arr = frame[name].to_numpy()
        else:
            arr = np.full(len(frame), "", dtype=object)
        if len(arr) == 0:
            return np.full(len(idx), "", dtype=object)
        return arr[np.clip(idx, 0, len(arr) - 1)]

    a_name = _safe(pick(s1, "name_core", s1_rows))
    b_name = _safe(pick(chunk, "name_core", cand_rows))
    a_addr = _safe(pick(s1, "address_norm", s1_rows))
    b_addr = _safe(pick(chunk, "address_norm", cand_rows))
    a_city = _safe(pick(s1, "city_norm", s1_rows))
    b_city = _safe(pick(chunk, "city_norm", cand_rows))
    a_state = _safe(pick(s1, "state_norm", s1_rows))
    b_state = _safe(pick(chunk, "state_norm", cand_rows))
    a_country = _safe(pick(s1, "country_norm", s1_rows))
    b_country = _safe(pick(chunk, "country_norm", cand_rows))
    a_tok_col = col(s1, "name_tokens")
    b_tok_col = col(chunk, "name_tokens")
    a_toks = [a_tok_col[i] for i in s1_rows]
    b_toks = [b_tok_col[i] for i in cand_rows]
    # Full locality candidate sets. `city_norm` holds one best guess, which is
    # what the blocking key uses; the pair comparison needs the whole set,
    # because a free-text address rarely agrees with itself about which chunk is
    # the town.
    a_local = _locality_sets(col(s1, "locality_tokens")[s1_rows])
    b_local = _locality_sets(col(chunk, "locality_tokens")[cand_rows])

    out: Dict[str, object] = {}
    if s1_id_col in s1.columns:
        out["s1_entity_id"] = s1[s1_id_col].to_numpy()[s1_rows]
    if cand_id_col in chunk.columns:
        out["source_record_id"] = chunk[cand_id_col].to_numpy()[cand_rows]
    for passthru in ("source_dataset", "matched_rules", "rule_bitmask"):
        if passthru in pairs.columns:
            out[passthru] = pairs[passthru].to_numpy()

    # ---- name ---------------------------------------------------------------
    tsr = _masked(_pairwise_ratio(a_name, b_name, fuzz.token_set_ratio, 0.01), a_name, b_name)
    sor = _masked(_pairwise_ratio(a_name, b_name, fuzz.token_sort_ratio, 0.01), a_name, b_name)
    jw = _masked(
        _pairwise_ratio(a_name, b_name, JaroWinkler.normalized_similarity), a_name, b_name
    )
    overlap = np.asarray([_token_overlap(x, y) for x, y in zip(a_toks, b_toks)], dtype=np.float64)
    stack = np.vstack([tsr, sor, jw, overlap])
    # Mean over the name metrics that exist for this pair. All four are on the
    # same 0-1 scale, so the mean is meaningful; NaN rows drop out rather than
    # dragging the average to zero.
    out["name_token_set_ratio"] = tsr
    out["name_token_sort_ratio"] = sor
    out["name_jaro_winkler"] = jw
    out["name_token_overlap_ratio"] = overlap
    out["name_similarity"] = _nanmean(stack.T, axis=1)

    # ---- address ------------------------------------------------------------
    out["address_similarity"] = _masked(
        _pairwise_ratio(a_addr, b_addr, fuzz.token_set_ratio, 0.01), a_addr, b_addr
    )

    # ---- extracted city / state --------------------------------------------
    # `city_similarity` compares the two best-guess cities (vectorized);
    # `city_exact` is a *set* intersection over the full locality candidates, so
    # "Miryalguda Mandal" on one side and "Miryalguda" on the other -- the same
    # place, written differently -- counts as agreement rather than a conflict.
    # Equality on one arbitrarily chosen token was the single biggest source of
    # phantom city conflicts in an earlier version of this.
    out["city_similarity"] = _masked(
        _pairwise_ratio(a_city, b_city, fuzz.token_set_ratio, 0.01), a_city, b_city
    )
    out["state_similarity"] = _masked(
        _pairwise_ratio(a_state, b_state, fuzz.token_set_ratio, 0.01), a_state, b_state
    )
    city_both = np.asarray(
        [(bool(x) and bool(y)) for x, y in zip(a_local, b_local)], dtype=bool
    ) if len(a_local) else np.zeros(0, dtype=bool)
    if not len(a_local):
        # No locality column: fall back to the single-token comparison.
        city_both = (a_city != "") & (b_city != "")
        city_exact = np.where(city_both, (a_city == b_city).astype(float), np.nan)
    else:
        city_exact = np.asarray(
            [
                (1.0 if not x.isdisjoint(y) else 0.0) if city_both[i] else np.nan
                for i, (x, y) in enumerate(zip(a_local, b_local))
            ],
            dtype=np.float64,
        )
    state_both = (a_state != "") & (b_state != "")
    # 1.0 equal, 0.0 both present and different, NaN when either side is empty:
    # "no state anywhere" is not evidence of a different state.
    out["city_exact"] = city_exact
    out["state_exact"] = np.where(state_both, (a_state == b_state).astype(float), np.nan)
    country_both = (a_country != "") & (b_country != "")
    out["country_exact"] = np.where(country_both, (a_country == b_country).astype(float), np.nan)

    # ---- phonetic agreement -------------------------------------------------
    from .block import phonetic_code

    scheme = "nysiis"
    a_code = _memoize(lambda v: phonetic_code(v, scheme), a_name)
    b_code = _memoize(lambda v: phonetic_code(v, scheme), b_name)
    code_both = np.asarray(
        [(a_code[i] != "" and b_code[i] != "") for i in range(len(a_code))], dtype=bool
    )
    out["phonetic_name_match"] = np.where(
        code_both, (a_code == b_code).astype(float), np.nan
    )

    # ---- character-level and domain features -------------------------------
    # Everything above is token- or whole-string-oriented, which is blind to the
    # two failure modes this dataset is full of: transposed characters, and a
    # name that is a bare domain. Both are cheap to add and neither is available
    # anywhere else in the feature set.
    from rapidfuzz.distance import Levenshtein

    n_pairs = len(a_name)
    if n_pairs == 0:
        return out

    def _lev_ratio(x: str, y: str) -> float:
        if not x or not y:
            return np.nan
        longest = max(len(x), len(y))
        return 1.0 - (Levenshtein.distance(x, y) / float(longest))

    # Character-level Jaro-Winkler on the raw name. Token metrics score
    # "Acme Bike" vs "Acme Bicycle" as merely overlapping; char-level catches
    # the shared stem, and it is also insensitive to word order.
    out["name_char_jaro_winkler"] = np.array(
        [JaroWinkler.similarity(a_name[i], b_name[i]) if a_name[i] and b_name[i] else np.nan
         for i in range(n_pairs)],
        dtype=float,
    )
    out["name_levenshtein_ratio"] = np.array(
        [_lev_ratio(a_name[i], b_name[i]) for i in range(n_pairs)], dtype=float
    )
    # Character-level agreement on the address, which suffers the same
    # abbreviation noise as the name ("ST"/"Street", "RD"/"Road").
    out["address_char_jaro_winkler"] = np.array(
        [JaroWinkler.similarity(a_addr[i], b_addr[i]) if a_addr[i] and b_addr[i] else np.nan
         for i in range(n_pairs)],
        dtype=float,
    )

    # Length and token-count deltas. A large gap with high token overlap is the
    # signature of a truncated listing ("Acme Bike" for "Acme Bicycle Company"),
    # which the ratio metrics alone read as weak evidence.
    def _safe_len(arr: np.ndarray) -> np.ndarray:
        return np.array([len(x) for x in arr], dtype=float)

    la, lb = _safe_len(a_name), _safe_len(b_name)
    out["name_len_ratio"] = np.where(
        (la > 0) & (lb > 0), np.minimum(la, lb) / np.maximum(la, lb), np.nan
    )
    ta = np.array([len(t) for t in a_toks], dtype=float)
    tb = np.array([len(t) for t in b_toks], dtype=float)
    # Expressed as a ratio, not an absolute count: ER_FEATURE_COLUMNS is
    # documented as 0-1 or NaN, and a raw count would break that contract and
    # let one long name dominate a tree split.
    out["name_token_count_ratio"] = np.where(
        (ta > 0) & (tb > 0), np.minimum(ta, tb) / np.maximum(ta, tb), np.nan
    )

    # ---- domain / website, mined from the name ------------------------------
    # There is no website column in this dataset, but the name frequently *is* one
    # ("associatedsoftwareproducts.com", "wilfordhancock.com"). A domain that
    # matches exactly is close to decisive, because two unrelated businesses
    # rarely claim the same domain. Derived only from `business_name`, so it is
    # available at prediction time on every source.
    _DOMAIN_RE = re.compile(r"\b((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})\b")

    def _domain(name: str) -> str:
        if not name:
            return ""
        m = _DOMAIN_RE.search(name.lower())
        if not m:
            return ""
        host = m.group(1)
        # Strip a leading "www."; it is never distinguishing.
        if host.startswith("www."):
            host = host[4:]
        # Only keep real-looking business domains. A bare "co.in"/"com" tail on a
        # non-domain token is noise, so require at least one label plus a
        # plausible TLD and a host of reasonable length.
        if len(host) < 5 or "." not in host:
            return ""
        return host

    a_dom = np.array([_domain(a_name[i]) for i in range(n_pairs)], dtype=object)
    b_dom = np.array([_domain(b_name[i]) for i in range(n_pairs)], dtype=object)
    dom_both = (a_dom != "") & (b_dom != "")
    out["domain_exact"] = np.where(dom_both, (a_dom == b_dom).astype(float), np.nan)
    out["domain_present_either"] = ((a_dom != "") | (b_dom != "")).astype(float)
    # Registrable-ish comparison: same last two labels, so "shop.acme.com" and
    # "acme.com" still register as related.
    out["domain_suffix_match"] = np.where(
        dom_both,
        np.array([
            1.0 if (".".join(a_dom[i].split(".")[-2:]) == ".".join(b_dom[i].split(".")[-2:]))
            else 0.0
            for i in range(n_pairs)
        ], dtype=float),
        np.nan,
    )

    # ---- script-aware name comparison ---------------------------------------
    # 11.3% of true Indian pairs pair a Devanagari name with a Latin one
    # ("Golden Gold Services" vs "गोल्डन गोल्ड सर्विसेज"). Whole-string metrics
    # score those near zero and the single phonetic flag cannot help because
    # Soundex/NYSIIS are defined only over ASCII. Comparing the *Latin* token
    # subset of each name recovers signal whenever either side carries any
    # Latin script, which includes mixed-script names.
    _LATIN_RE = re.compile(r"^[a-z0-9]+$")

    def _latin_toks(name: str) -> frozenset:
        if not name:
            return frozenset()
        return frozenset(t for t in name.split() if _LATIN_RE.match(t))

    a_lat = [_latin_toks(a_name[i]) for i in range(n_pairs)]
    b_lat = [_latin_toks(b_name[i]) for i in range(n_pairs)]
    out["name_latin_token_jaccard"] = np.array(
        [(len(a_lat[i] & b_lat[i]) / len(a_lat[i] | b_lat[i]))
         if (a_lat[i] and b_lat[i]) else np.nan for i in range(n_pairs)],
        dtype=float,
    )
    out["name_cross_script"] = np.where(
        (a_name != "") & (b_name != ""),
        (_has_non_latin(a_name) != _has_non_latin(b_name)).astype(float),
        np.nan,
    )

    # ---- missingness --------------------------------------------------------
    def flag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a | b).astype(float)

    out["name_missing_either"] = flag(
        boolcol(s1, "name_missing")[s1_rows], boolcol(chunk, "name_missing")[cand_rows]
    )
    out["name_placeholder_either"] = flag(
        boolcol(s1, "name_placeholder")[s1_rows], boolcol(chunk, "name_placeholder")[cand_rows]
    )
    out["address_missing_either"] = flag(
        boolcol(s1, "address_missing")[s1_rows], boolcol(chunk, "address_missing")[cand_rows]
    )
    out["address_parse_failed_either"] = flag(
        boolcol(s1, "address_parse_failed")[s1_rows],
        boolcol(chunk, "address_parse_failed")[cand_rows],
    )
    out["city_missing_either"] = flag(
        boolcol(s1, "city_missing")[s1_rows], boolcol(chunk, "city_missing")[cand_rows]
    )
    out["state_missing_either"] = flag(
        boolcol(s1, "state_missing")[s1_rows], boolcol(chunk, "state_missing")[cand_rows]
    )
    out["country_missing_either"] = flag(
        boolcol(s1, "country_missing")[s1_rows], boolcol(chunk, "country_missing")[cand_rows]
    )
    n_missing = np.zeros(len(s1_rows), dtype=float)
    for fld in ("name", "address", "city", "state", "country"):
        n_missing += out[f"{fld}_missing_either"].astype(float)
    out["n_fields_missing_total"] = n_missing

    # ---- blocking provenance ------------------------------------------------
    # Same defensive coercion as the positions: concatenating two batches
    # promotes an all-empty int column to object, and a bitwise AND on an object
    # array raises deep inside this function instead of here.
    if "rule_bitmask" in pairs.columns:
        mask = pd.to_numeric(pairs["rule_bitmask"], errors="coerce").fillna(0)
        mask = mask.to_numpy(dtype=np.int64)
    else:
        mask = np.zeros(len(pairs), dtype=np.int64)
    from .block import BLOCK_RULES, FALLBACK_FUZZY, FALLBACK_NEIGHBORHOOD, _rule_bit

    for rule in ER_BLOCKING_RULES:
        out[f"blk_{rule}"] = ((mask & _rule_bit(rule)) != 0).astype(float)
    out["n_rules_matched"] = (
        np.asarray([bin(int(m)).count("1") for m in mask], dtype=float)
        if len(mask) else np.zeros(0, dtype=float)
    )
    # A pair that exists only because the fallback pass produced it is weaker
    # evidence by construction, and the model should be able to learn that.
    out["blk_fallback"] = (
        (mask & (_rule_bit(FALLBACK_FUZZY) | _rule_bit(FALLBACK_NEIGHBORHOOD))) != 0
    ).astype(float)

    # ---- original text, for review and output only --------------------------
    if include_raw:
        # Both spellings are accepted: `business_name` on a raw frame (tests,
        # one-off scripts) and `name_raw` on a cleaned one, which is what the
        # pipeline passes. Looking for one and silently emitting nothing for the
        # other is how a review file ends up with empty address columns.
        for side, frame, rows in (("s1", s1, s1_rows), ("cand", chunk, cand_rows)):
            for candidates, out_col in (
                (("business_name", "name_raw"), f"{side}_business_name_raw"),
                (("business_address", "address_raw"), f"{side}_business_address_raw"),
                (("country", "country_raw"), f"{side}_country_raw"),
            ):
                for raw_col in candidates:
                    if raw_col in frame.columns:
                        out[out_col] = (
                            frame[raw_col].fillna("").astype(str).to_numpy()[rows]
                        )
                        break
    return pd.DataFrame(out)


@dataclass
class CompositeWeights:
    """Explainable score weights (Change 6). Defaults come from the spec.

    They are a *reporting* device, not the decision rule: the LightGBM
    classifier remains the only thing that decides a match. The weights exist
    so a reviewer can see, for a rejected pair, which component was weak.
    """

    name: float = 0.45
    address: float = 0.30
    city_state: float = 0.15
    country: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {"name": self.name, "address": self.address,
                "city_state": self.city_state, "country": self.country}


def compute_weighted_composite_score(
    features: pd.DataFrame,
    weights: Optional[CompositeWeights] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    return_breakdown: bool = False,
):
    """Transparent weighted average of the Change 5 features, in [0, 1].

    Per group, the mean of the group's *available* (non-NaN) features; then the
    weighted sum over the groups that exist for this pair. When a group's
    features are all missing its weight is redistributed proportionally across
    the remaining groups, so a pair with no address at all is not punished for
    it -- the score still lands on 0-1 and stays comparable across pairs.

    Returns an ndarray (or ``(scores, breakdown_frame)`` with
    ``return_breakdown=True``). Never a decision: see `assemble.decide_match_status`.
    """
    weights = weights or CompositeWeights()
    groups = groups or ER_COMPOSITE_GROUPS
    n = len(features)
    if n == 0:
        return (np.zeros(0, dtype=float), pd.DataFrame()) if return_breakdown else np.zeros(0)

    group_scores: Dict[str, np.ndarray] = {}
    for name, cols in groups.items():
        present = [c for c in cols if c in features.columns]
        if not present:
            group_scores[name] = np.full(n, np.nan)
            continue
        block = features[present].to_numpy(dtype=float)
        # block is (n_pairs, n_features): reduce over the features.
        group_scores[name] = _nanmean(block, axis=1)

    w = weights.as_dict()
    # matrix is (n_groups, n_pairs): one row per group so the group weights
    # broadcast as a column.
    matrix = np.vstack([group_scores[g] for g in groups])
    wvec = np.asarray([w.get(g, 0.0) for g in groups], dtype=float)[:, None]
    available = ~np.isnan(matrix)

    # Auto-rebalance: for each pair, renormalize the weights over the groups
    # that actually have evidence.
    wmat = np.where(available, wvec, 0.0)
    denom = wmat.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_w = np.where(denom > 0, wmat / denom, 0.0)
    filled = np.where(np.isnan(matrix), 0.0, matrix)
    scores = np.nansum(filled * norm_w, axis=0)
    # Nothing at all was available for this pair: report "no evidence", not 0.
    scores = np.where(denom > 0, scores, np.nan)

    if not return_breakdown:
        return scores
    breakdown = pd.DataFrame({f"wcs_{g}": group_scores[g] for g in groups})
    breakdown["weighted_composite_score"] = scores
    return scores, breakdown

