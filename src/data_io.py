"""Reading and writing the challenge TSVs, and the normalized record table.

Everything downstream works on a compact numeric representation, never on the
raw strings. At 11.7M test records (1.73M S1 + 4.89M S2 + 5.08M S3) holding
Python string objects for every field costs several GB and makes the pairwise
work impossible. So each record is reduced once, to codes and small ints, and
the strings are dropped.

The normalization is deliberately country-agnostic. Measured on the training
ground truth, the facts that constrain the design are:

* **Country never disagrees** between a Source 1 record and any of its true
  matches (0 mismatches in 152,302 sampled pairs). It is therefore a safe
  blocking key -- and it is also the *only* key that keeps working for France,
  which is 15% of the test set, absent from training, and has no postal codes.
  Any logic keyed on US zip codes silently scores zero on France.

* **France has no postcodes at all.** Addresses look like
  ``175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine``.
  There is nothing to extract, so the parser must degrade to "tokens", not fail.

* **India names are in Devanagari** (5.4% of S2, 3.0% of S3) and 11.3% of true
  Indian pairs are *cross-script*: ``Golden Gold Services`` vs
  ``गोल्डन गोल्ड सर्विसेज``. An ASCII-only normalizer folds those to the empty
  string, which is a silent total failure on 11% of Indian pairs. So the
  normalizer must be Unicode-aware and must *never* discard a script wholesale.
  Where a name is mixed-script ("Golden गोल्ड सर्विसेज") the Latin tokens are
  kept separately, because that is often the only signal shared with the Latin
  spelling in Source 1.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

csv.field_size_limit(min(sys.maxsize, 10**8))

# ---------------------------------------------------------------------------
# Unicode handling
# ---------------------------------------------------------------------------

# Scripts we must NOT destroy. Devanagari is the important one, but the rule is
# general: fold accents off Latin, keep every other script's letters intact.
_COMBINING = dict.fromkeys(
    [0x0300 + i for i in range(0x10F)]
    + [0x1AB0 + i for i in range(0x260)]
    + [0x1DC0 + i for i in range(0x100)]
    + [0x20D0 + i for i in range(0x30)]
    + [0xFE20 + i for i in range(0x20)],
    None,
)

# A token is "wordlike" if it contains a letter from any script.
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_DIGIT_RE = re.compile(r"\d+")


def strip_accents(text: str) -> str:
    """Fold Latin accents only. Devanagari matras are left alone.

    Using a general NFKD strip here would silently decompose and then discard
    Devanagari vowel signs, corrupting the one script we most need intact.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return decomposed.translate(_COMBINING)


def squash(text: Optional[str]) -> str:
    """Lowercase, fold Latin accents, collapse every non-word run to a space.

    Unicode-aware: ``\\W`` is script-agnostic in Python's ``re`` with the default
    (str) behaviour, so Devanagari and accented Latin both survive.
    """
    if not text:
        return ""
    text = strip_accents(str(text)).lower()
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def split_scripts(text: str) -> Tuple[List[str], List[str]]:
    """Return ``(latin_tokens, non_latin_tokens)``.

    Mixed-script names are common in the Indian data
    (``Golden गोल्ड सर्विसेज``). The Latin tokens are the part most likely to
    appear in Source 1's Latin spelling, so they are kept as their own list
    rather than being mixed into a bag of tokens where a downstream key would
    drown them.
    """
    latin, other = [], []
    for tok in squash(text).split():
        if _LATIN_LETTER_RE.search(tok):
            latin.append(tok)
        elif _LETTER_RE.search(tok):
            other.append(tok)
    return latin, other


# ---------------------------------------------------------------------------
# Name cleaning
# ---------------------------------------------------------------------------

# Legal/entity suffixes across all three countries. Stripped only from the
# trailing end, and only for the "core" representation.
LEGAL_SUFFIXES = {
    # US / UK
    "llc", "l l c", "inc", "incorporated", "corp", "corporation", "co",
    "company", "ltd", "limited", "lp", "llp", "plc", "pc", "pa", "lc",
    # France: Societe forms
    "sarl", "s a r l", "sas", "s a s", "sasu", "eurl", "sas u", "snc", "scs",
    "scop", "sa", "s a", "sas s", "gi", "g i",
    # India
    "pvt", "private", "ltd", "limited", "pvt ltd", "and sons", "and co",
    "and company", "enterprises", "traders", "industries", "industries pvt ltd",
    "expo", "agency", "agencies", "store", "stores", "shop", "mart",
    "resto", "restos",
}

# Generic words that distinguish nothing. Stripped from name_core only.
GENERIC_WORDS = {
    "group", "holdings", "enterprises", "ventures", "partners", "services",
    "service", "systems", "supply", "supplies", "solutions", "international",
    "worldwide", "global", "national", "company", "co", "and", "the", "of",
    "business", "center", "centre", "bureau", "agency", "works", "house",
}

_DBA_RE = re.compile(r"\b(dba|d b a|aka|a k a|fka|f k a)\b")


def _strip_trailing(tokens: List[str], vocab) -> List[str]:
    out = list(tokens)
    while out and out[-1] in vocab:
        out.pop()
    return out


def normalize_name(raw: Optional[str]) -> Tuple[str, str, List[str], List[str]]:
    """Return ``(name_norm, name_core, latin_tokens, other_script_tokens)``.

    ``name_norm``  squashed text, DBA marker resolved, legal suffix retained.
    ``name_core``  trailing legal suffixes and generic words removed, so
                   "Acme Bicycle Company LLC" and "Acme Bicycle" collide.
    ``latin_tokens`` / ``other_script_tokens``  the script split, which is what
                   makes mixed-script Indian names matchable at all.
    """
    if not raw:
        return "", "", [], []
    text = strip_accents(str(raw)).lower()
    m = _DBA_RE.search(text)
    if m:
        text = text[: m.start()]
    tokens = [t for t in squash(text).split() if t]
    if not tokens:
        return "", "", [], []

    name_norm = " ".join(tokens)
    core = _strip_trailing(tokens, LEGAL_SUFFIXES)
    core = _strip_trailing(core, GENERIC_WORDS)
    if core and core[0] in ("the",):
        core = core[1:]
    # If stripping consumed the whole name, keep the unstripped form: a short
    # name is still much better signal than nothing.
    name_core = " ".join(core) if core else name_norm
    latin, other = split_scripts(name_norm)
    return name_norm, name_core, latin, other


# ---------------------------------------------------------------------------
# Address handling (country-agnostic)
# ---------------------------------------------------------------------------

# Kept for the US subset, where it is genuinely useful, but it is no longer the
# primary path: France has no street numbers or postcodes to speak of, and
# Indian addresses are building/landmark style ("H.No.16-11-23/37/A, 2Nd Floor,
# Opp.Rta Office").
STREET_SUFFIXES = {
    "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "aven": "avenue", "avenue": "avenue",
    "rd": "road", "road": "road",
    "blvd": "boulevard", "boulevard": "boulevard", "blv": "boulevard",
    "dr": "drive", "drive": "drive", "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court", "pl": "place", "place": "place",
    "way": "way", "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    # French
    "r": "rue", "rue": "rue", "bd": "boulevard", "av": "avenue",
    "pl": "place", "imp": "impasse", "allee": "allee", "allée": "allee",
    "chem": "chemin", "quai": "quai", "cours": "cours", "square": "square",
    "sq": "square", "villa": "villa", "cite": "cite", "cit": "cite",
}

_UNIT_WORDS = {
    "apt", "apartment", "ste", "suite", "unit", "rm", "room", "fl", "floor",
    "bldg", "building", "lot", "no", "num", "dept", "trlr",
    # India
    "flat", "door", "h no", "hno", "shop", "shop no", "grd", "ground",
    "first", "second", "third", "st floor",
}

_NUM_RE = re.compile(r"^\d+[a-z]?$")
_ZIP_RE = re.compile(r"\b(\d{5,6})\b")

# Words that are pure address scaffolding and carry no locality. Removing them
# before building an address key stops "Near SBI ATM" from being a key.
_STOPWORDS = {
    "near", "opp", "opposite", "behind", "beside", "next", "adjacent", "cross",
    "road", "street", "avenue", "floor", "flat", "door", "no", "towards",
    "landmark", "building", "sector", "phase", "block", "plot", "house",
    "the", "and", "of", "at", "by", "in", "new", "old", "via",
}


def parse_address(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """Best-effort country-agnostic address parse.

    Deliberately returns whatever it can rather than assuming a US layout:

    ``zip``        5-6 digit token anywhere (US zip / Indian PIN). None for
                   France, which has none -- and that is fine, callers treat
                   None as "this country has no postcodes".
    ``number``     leading house/plot number, if the address starts with one.
    ``street_core``street-ish tokens with suffixes canonicalized and
                   scaffolding words removed.
    ``city``       best-effort: a comma-delimited chunk, else the last token
                   that is not a zip/region.
    ``tokens``     the full squashed token list -- the fallback key material.
    """
    out: Dict[str, Optional[object]] = {
        "zip": None, "number": None, "street_core": None,
        "city": None, "region": None, "tokens": [],
    }
    if not raw:
        return out
    text = strip_accents(str(raw)).lower()
    chunks = [squash(c) for c in str(raw).split(",")]
    chunks = [c for c in chunks if c]
    toks = [t for t in squash(text).split() if t]
    out["tokens"] = toks
    if not toks:
        return out

    # --- postal code -------------------------------------------------------
    for t in toks:
        if _ZIP_RE.fullmatch(t):
            out["zip"] = t
            break
    if out["zip"] is None:
        m = _ZIP_RE.search(text)
        if m:
            out["zip"] = m.group(1)

    # --- house / plot number ----------------------------------------------
    if toks and _NUM_RE.match(toks[0]):
        out["number"] = toks[0]

    # --- street ------------------------------------------------------------
    # Take the first comma chunk when present (US/France put the street line
    # first); otherwise use the leading run. Then canonicalize suffixes and drop
    # scaffolding so "Near SBI ATM" contributes nothing.
    head = chunks[0].split() if chunks else toks
    street_toks: List[str] = []
    for t in head:
        if t in _STOPWORDS or t in _UNIT_WORDS:
            continue
        street_toks.append(STREET_SUFFIXES.get(t, t))
    if not street_toks:
        street_toks = [t for t in head if t not in _STOPWORDS]
    out["street_core"] = " ".join(street_toks[:6]) if street_toks else None

    # --- city / region -----------------------------------------------------
    # With commas, the chunk before a trailing zip/region chunk is the city.
    tail = chunks[1:] if len(chunks) > 1 else []
    picked_city = None
    picked_region = None
    for chunk in reversed(tail):
        ct = chunk.split()
        if not ct:
            continue
        if picked_region is None and (out["zip"] and out["zip"] in ct or len(ct) <= 2):
            # Ambiguous between city and region; a 1-2 token tail in the FR/IN
            # style is the region ("Hauts-de-France", "Telangana").
            if picked_city is not None:
                picked_region = chunk
                continue
        if picked_city is None:
            picked_city = chunk
    if picked_city is None and len(tail) == 0:
        # No commas: the last non-zip token is a poor but non-null city guess.
        cand = [t for t in toks if t != out["zip"]]
        picked_city = cand[-1] if cand else None
    out["city"] = picked_city
    out["region"] = picked_region
    return out


# ---------------------------------------------------------------------------
# Feature extraction -> compact record table
# ---------------------------------------------------------------------------

# Columns produced by `build_record_frame`. All are compact: codes or small
# ints, never object strings. This is what makes 11.7M records tractable.
RECORD_COLUMNS = [
    "eid", "country", "name_norm", "name_core", "name_prefix4",
    "name_soundex", "name_latin", "name_other",
    "addr_zip", "addr_zip3", "addr_number", "addr_street", "addr_city",
    "addr_tok", "addr_tok_count",
]

# Punctuation/decorations that must not become key material.
_NOISE_PREFIX_RE = re.compile(r"^[<>{}\[\]\-_=+*/\\|~^`\"'.,;:!?()\[\]{}‘’“”]+")


def _code(series: pd.Series) -> pd.Series:
    """Factorize to int32 codes with -1 for missing."""
    return pd.factorize(series, sort=True)[0].astype(np.int32)


def soundex(text: str) -> str:
    """Classic 4-character Soundex over Latin tokens only.

    Non-Latin tokens are skipped: Soundex is defined on ASCII letters, and
    feeding it Devanagari produces empty codes that would collide across
    unrelated names.
    """
    latin = [t for t in squash(text).split() if _LATIN_LETTER_RE.search(t)]
    if not latin:
        return ""
    word = latin[0].upper()
    codes = {
        **{c: "1" for c in "BFPV"}, **{c: "2" for c in "CGJKQSXZ"},
        **{c: "3" for c in "DT"}, "L": "4", **{c: "5" for c in "MN"},
        "R": "6",
    }
    out = [word[0]]
    prev = codes.get(word[0], "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            out.append(code)
        if ch not in "HW":
            prev = code
        if len(out) == 4:
            break
    return ("".join(out) + "000")[:4]


def build_record_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw source frame into the compact record table.

    ``df`` must have columns entity_id, business_name, business_address,
    country. Returns a frame with RECORD_COLUMNS, where the string-ish columns
    are pandas categoricals (dictionary-encoded, so 11.7M distinct addresses cost
    a code array plus one dictionary, not 11.7M Python strings).
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in RECORD_COLUMNS})

    names = [normalize_name(v) for v in df["business_name"].fillna("")]
    addrs = [parse_address(v) for v in df["business_address"].fillna("")]

    out = pd.DataFrame(index=range(n))
    out["eid"] = df["entity_id"].astype("category")
    out["country"] = _code(df["country"].fillna("").astype(str))

    out["name_norm"] = _code(pd.Series([a[0] for a in names], index=out.index))
    out["name_core"] = _code(pd.Series([a[1] for a in names], index=out.index))
    out["name_latin"] = _code(pd.Series([" ".join(a[2]) for a in names], index=out.index))
    out["name_other"] = _code(pd.Series([" ".join(a[3]) for a in names], index=out.index))
    out["name_prefix4"] = _code(pd.Series([a[1][:4] for a in names], index=out.index))
    out["name_prefix8"] = _code(pd.Series([a[1][:8] for a in names], index=out.index))
    out["name_soundex"] = _code(pd.Series([soundex(a[0]) for a in names], index=out.index))

    out["addr_zip"] = _code(pd.Series([a["zip"] or "" for a in addrs], index=out.index))
    out["addr_zip3"] = _code(
        pd.Series([(a["zip"] or "")[:3] for a in addrs], index=out.index)
    )
    out["addr_number"] = _code(pd.Series([a["number"] or "" for a in addrs], index=out.index))
    out["addr_street"] = _code(pd.Series([a["street_core"] or "" for a in addrs], index=out.index))
    out["addr_city"] = _code(pd.Series([a["city"] or "" for a in addrs], index=out.index))
    out["addr_tok"] = _code(pd.Series([" ".join(a["tokens"][:8]) for a in addrs], index=out.index))
    out["addr_tok_count"] = np.minimum(
        np.array([len(a["tokens"]) for a in addrs], dtype=np.int32), 32
    )
    return out
