"""Step 1 -- data cleaning & normalization.

Every field in every source is reduced to a *comparable* representation before
any comparison happens. The guiding rule is that different sources express the
same fact in different surface forms:

    "The Acme Bicycle Co., LLC"   Source 1 legal name
    "ACME BICYCLE COMPANY LLC"    Source 2 registry name
    "Acme Bike Shop"              Source 3 marketing name

    "123 North Main St., Ste. 200"  Source 1
    "123 N MAIN ST STE 200"         Source 2
    "123 N Main St"                 Source 3

All three must land on the same normalized tokens, or every downstream
similarity score is measuring formatting instead of identity.

Design decisions
----------------
* **Nothing is dropped.** A record missing its phone still has a name and an
  address. Instead of filtering it out we normalize what exists and raise an
  explicit ``*_missing`` boolean for each field. Those indicators are fed to
  the model (see features.py) because "phone is missing" is *itself* evidence
  -- a pair where both phones are missing is far weaker than a pair where one
  is present and matches.

* **Addresses are parsed, not string-matched.** A single normalized address
  string is not enough: "123 N Main St Springfield IL 62704" and "123 N Main St
  Unit 4 Springfield IL 62704" should agree on street/city/zip and disagree
  only on the unit. That yields far more usable signal than one fuzzy score.

* **Two name representations are kept.** ``name_norm`` keeps the full cleaned
  name (including the category-ish words), while ``name_core`` additionally
  strips legal suffixes *and* generic business words ("shop", "company",
  "services"). "Acme Bicycle Company" and "Acme Bike Shop" share almost nothing
  on ``name_norm`` but collide on ``name_core``. The model sees both, so it can
  learn how much to trust each.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Legal / organizational suffixes
# ---------------------------------------------------------------------------

# Trailing tokens that carry no distinguishing information. Note we include the
# generic business words here because a marketing name like "Acme Bike Shop"
# and a legal name like "Acme Bicycle Company" are the same entity, and the
# shared signal is "acme bicycle"-ish, not "shop".
LEGAL_SUFFIXES = {
    # entity form
    "llc", "l l c", "lc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "limited", "lp", "llp", "plc", "pc", "pa",
    "gmbh", "ag", "bv", "nv", "sa", "srl", "spa", "pte", "pty",
    # generic business words (stripped only for name_core)
    "shop", "store", "inc", "the", "and", "of", "group", "holdings",
    "enterprises", "ventures", "partners", "services", "service", "systems",
    "supply", "supplies", "solution", "solutions", "international",
    "worldwide", "global", "national", "usa", "us", "inc",
}

# Words removed from name_core but KEPT in name_norm.
GENERIC_BUSINESS_WORDS = {
    "shop", "store", "group", "holdings", "enterprises", "ventures",
    "partners", "services", "service", "systems", "supply", "supplies",
    "solution", "solutions", "international", "worldwide", "global",
    "national", "company", "co",
}

# "dba" / "aka" / "fka" introduce an alternate name. When these appear we take
# the part BEFORE the marker as the primary name -- the leading name is the
# registered one and is far more likely to appear in the other sources.
DBA_MARKERS = {"dba", "aka", "fka", "a k a", "d b a"}

# ---------------------------------------------------------------------------
# Street suffix canonicalization
# ---------------------------------------------------------------------------

# USPS abbreviations -> canonical long form. Applied to the *last* recognized
# suffix token so "St" and "Street" become identical.
STREET_SUFFIXES = {
    "st": "street", "str": "street", "stree": "street",
    "ave": "avenue", "av": "avenue", "aven": "avenue",
    "blvd": "boulevard", "blv": "boulevard", "boul": "boulevard",
    "rd": "road",
    "dr": "drive", "drv": "drive",
    "ln": "lane",
    "ct": "court", "crt": "court",
    "pl": "place",
    "plz": "plaza",
    "sq": "square",
    "ter": "terrace", "terr": "terrace",
    "cir": "circle", "circ": "circle",
    "hwy": "highway", "highwy": "highway",
    "pkwy": "parkway", "pky": "parkway", "parky": "parkway",
    "expy": "expressway",
    "fwy": "freeway",
    "aly": "alley",
    "bnd": "bend", "bnds": "bends",
    "xing": "crossing",
    "hvn": "haven",
    "mnr": "manor",
    "mt": "mount",
    "ft": "fort",
    "vw": "view",
    "trl": "trail",
    "wy": "way",
}

DIRECTIONS = {
    "n": "n", "north": "n", "s": "s", "south": "s",
    "e": "e", "east": "e", "w": "w", "west": "w",
    "ne": "ne", "northeast": "ne", "nw": "nw", "northwest": "nw",
    "se": "se", "southeast": "se", "sw": "sw", "southwest": "sw",
}

# The canonical long forms must also be keys, otherwise "123 Main Street" does
# not recognise its own suffix while "123 Main St" does. Adding identity
# mappings makes the vocabulary closed under its own output.
STREET_SUFFIXES.update({long: long for long in set(STREET_SUFFIXES.values())})

# Secondary-unit designators. These are stripped out of the street name so that
# "Ste 200" does not pollute the street-name comparison.
UNIT_DESIGNATORS = {
    "apt", "apartment", "ste", "suite", "unit", "rm", "room", "fl", "floor",
    "bldg", "building", "#", "no", "num", "lot", "dept", "trlr", "rte",
    "box", "po", "pobox",
}

# ---------------------------------------------------------------------------
# State / category vocabularies
# ---------------------------------------------------------------------------

STATE_ABBREV = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}

# Crosswalk between the controlled vocabularies the three sources use.
#
# This table is the single source of truth for category equivalence. The model
# gets a `category_match` flag from it *in addition to* raw string similarity, so
# "Restaurants" and "Eating Places & Cafes" count as agreeing even though they
# share no characters.
#
# Each canonical value is a business *type*, and each type has several surface
# forms drawn from the vocabularies the sources actually publish (Source 1/2
# plain English, Source 3 directory-ese). The synthetic generator derives its
# per-source vocabularies from this same dict, so the benchmark can never drift
# out of sync with the normalizer.
CATEGORY_SYNONYMS = {
    # --- food ---------------------------------------------------------------
    "restaurant": "restaurant", "restaurants": "restaurant",
    "eating places": "restaurant", "food services": "restaurant",
    "cafe": "restaurant", "cafes": "restaurant",
    "coffee": "coffee", "coffee shop": "coffee", "coffeehouse": "coffee",
    "coffee and tea shops": "coffee",
    "bakery": "bakery", "bakeries": "bakery",
    "catering": "catering", "caterers": "catering",
    # --- automotive ---------------------------------------------------------
    "auto": "auto", "auto repair": "auto", "automotive": "auto",
    "auto services": "auto", "auto repair shops": "auto",
    "mechanical": "auto", "car repair": "auto", "tire": "auto",
    "tires": "auto", "body shop": "auto",
    "bike": "bike", "bicycle": "bike", "bicycle dealers": "bike",
    "bike shops": "bike", "cycling": "bike",
    # --- home / building trades --------------------------------------------
    "hardware": "hardware", "hardware store": "hardware",
    "building supply": "hardware", "home improvement": "hardware",
    "plumbing": "plumbing", "plumber": "plumbing", "plumbers": "plumbing",
    "plumbing contractors": "plumbing",
    "electrical": "electrical", "electrician": "electrical",
    "electrical contractors": "electrical",
    "electric": "electric", "electricians": "electric",
    "construction": "construction", "contractor": "construction",
    "general contractor": "construction", "builders": "construction",
    "roofing": "roofing", "roofing contractors": "roofing", "roofer": "roofing",
    "siding": "siding", "siding contractors": "siding",
    "pool": "pool", "pool service": "pool", "pool and spa services": "pool",
    "landscaping": "landscaping", "lawn care": "landscaping",
    "landscaping services": "landscaping", "lawn maintenance": "landscaping",
    "gardener": "landscaping", "pest control": "landscaping",
    "exterminator": "landscaping", "locksmith": "landscaping",
    "moving": "moving", "moving company": "moving", "movers": "moving",
    "storage": "storage", "self storage": "storage", "storage units": "storage",
    "cleaning": "cleaning", "dry cleaner": "cleaning",
    "dry cleaners and laundry": "cleaning", "cleaners": "cleaning",
    "laundry": "cleaning",
    # --- professional services ----------------------------------------------
    "legal": "legal", "law office": "legal", "legal services": "legal",
    "attorney": "legal", "law firm": "legal", "attorneys": "legal",
    "law firms": "legal",
    "accounting": "accounting", "accountant": "accounting",
    "accounting and tax services": "accounting", "bookkeeping": "accounting",
    "cpa": "accounting", "tax preparation": "accounting",
    "insurance": "insurance", "insurance agency": "insurance",
    "insurance agencies": "insurance",
    "realestate": "realestate", "real estate": "realestate",
    "realty": "realestate", "realtor": "realestate", "realtors": "realestate",
    "finance": "finance", "bank": "finance", "banking": "finance",
    "financial services": "finance",
    "printing": "printing", "print shop": "printing",
    "printing and copy centers": "printing", "sign shop": "printing",
    "travel": "travel", "travel agency": "travel", "tour": "travel",
    "photography": "photography", "photographer": "photography",
    # --- health -------------------------------------------------------------
    "dental": "dental", "dentist": "dental", "dental services": "dental",
    "dentists": "dental", "orthodontist": "dental",
    "medical": "medical", "doctor": "medical", "physician": "medical",
    "clinic": "medical", "hospital": "medical", "chiropractor": "medical",
    "pharmacy": "pharmacy", "pharmacies": "pharmacy",
    "pharmacies and drug stores": "pharmacy", "drugstore": "pharmacy",
    "optometry": "optometry", "optician": "optometry",
    "opticians and ophthalmologists": "optometry",
    "fitness": "fitness", "gym": "fitness", "health club": "fitness",
    "fitness centers and gyms": "fitness", "yoga": "fitness",
    "pilates": "fitness",
    # --- retail / personal --------------------------------------------------
    "hardware retail": "hardware", "furniture": "furniture",
    "furniture store": "furniture", "furniture stores": "furniture",
    "floral": "floral", "florist": "floral", "flower shop": "floral",
    "florists": "floral", "flowers": "floral",
    "retail": "retail", "clothing": "retail", "apparel": "retail",
    "boutique": "retail",
    "grocery": "grocery", "supermarket": "grocery", "food market": "grocery",
    "convenience store": "grocery", "liquor store": "grocery",
    "pet": "pet", "pet store": "pet", "pets": "pet", "pet stores": "pet",
    "pet supplies": "pet", "veterinarian": "pet", "vet": "pet",
    "animal hospital": "pet",
    "beauty": "beauty", "hair salon": "beauty", "beauty salon": "beauty",
    "beauty salons": "beauty", "barbershop": "beauty", "barber": "beauty",
    "barber shops": "beauty", "nail salon": "beauty",
    "personalcare": "personalcare", "tattoo": "personalcare",
    "massage": "personalcare", "day spa": "personalcare", "spa": "personalcare",
    "childcare": "childcare", "daycare": "childcare",
    "child care services": "childcare", "day care centers": "childcare",
    "hotel": "hotel", "hotels": "hotel", "motel": "hotel", "inn": "hotel",
}

# Postal codes, precomputed once so the state sniffer does not rebuild the set
# on every row.
_VALID_STATES = frozenset(STATE_ABBREV.values())

# ---------------------------------------------------------------------------
# Region vocabulary
#
# The training data covers the US and India; the test set adds France, which
# publishes no postcodes and whose regions are not needed for any decision here.
# This table exists to *recognise* a region token so it can be excluded from the
# locality candidates -- "Karnataka, Bangalore South" is one region and one
# city, not two competing cities.
#
# It is a gazetteer of administrative divisions, not a business registry.
# Nothing here identifies a business, and an unrecognised token simply stays a
# locality, which is the safe direction: a wrong region guess costs one weak
# feature, while treating a region as a city breaks the city comparison for
# every record in that country.
# ---------------------------------------------------------------------------
REGION_ABBREV: Dict[str, str] = dict(STATE_ABBREV)
REGION_ABBREV.update({
    # India -- states and union territories, standard abbreviations.
    "andhra pradesh": "AP", "arunachal pradesh": "AR", "assam": "AS",
    "bihar": "BR", "chhattisgarh": "CG", "goa": "GA", "gujarat": "GJ",
    "haryana": "HR", "himachal pradesh": "HP", "jharkhand": "JH",
    "karnataka": "KA", "kerala": "KL", "madhya pradesh": "MP",
    "maharashtra": "MH", "manipur": "MN", "meghalaya": "ML",
    "mizoram": "MZ", "nagaland": "NL", "odisha": "OD", "orissa": "OD",
    "punjab": "PB", "rajasthan": "RJ", "sikkim": "SK", "tamil nadu": "TN",
    "telangana": "TG", "tripura": "TR", "uttar pradesh": "UP",
    "uttarakhand": "UK", "west bengal": "WB", "delhi": "DL",
    "puducherry": "PY", "pondicherry": "PY", "lakshadweep": "LD",
})
_REGION_CODES = frozenset(REGION_ABBREV.values()) | frozenset(REGION_ABBREV)

# Address-chunk tokens that mark a chunk as *not* a place name: secondary-unit
# designators, plus the words that introduce a survey/plot/flat number.
_NON_LOCALITY_TOKENS = set(UNIT_DESIGNATORS) | {
    "survey", "plot", "premises", "property", "godown", "warehouse",
    "building", "block", "flat", "shop", "office", "floor", "level",
}

# Landmark references ("Near SBI ATM", "Opposite City Hospital") are a documented
# address form in this dataset. They are a *location hint*, never a place name, so
# a chunk that starts with one of these is not a locality candidate.
_LANDMARK_PREFIXES = frozenset({
    "near", "opposite", "opp", "behind", "beside", "next", "adjacent", "close",
    "besides", "alongside", "infront", "before",
})

_POSTCODE_TOKEN_RE = re.compile(r"^\d{4,6}(?:-\d{2,4})?$")
# "c-311", "311/2", "12a" -- a flat or door number, never a locality.
_FLAT_ID_RE = re.compile(r"^[a-z]?\s*\d+[a-z]?(?:\s*/\s*\d+[a-z]?)?$")

# Fields that participate in blocking, in priority order. Used to compute the
# missingness summaries.
KEY_FIELDS = ["name", "address", "city", "state", "zip", "phone", "category"]

_MISSING_TOKENS = {"", "n a", "na", "n/a", "none", "null", "nan", "unknown",
                   "not available", "-", "--", "?", "no data", "n.a."}

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# Matches a NON-digit, so `.sub("", s)` strips everything that is not a digit
# and leaves the digits behind. (The inverse -- a `\d` pattern -- would delete
# the digits we want to keep.)
_NON_DIGIT_RE = re.compile(r"\D")
_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_HOUSE_NUM_RE = re.compile(r"^(\d+[a-z]?)\s+(.*)$", re.S)
_POBOX_RE = re.compile(r"\b(?:p\.?\s*o\.?\s*)?box\s+(\w+)\b", re.I)
_UNIT_RE = re.compile(
    r"(?:#|apt\.?|apartment|ste\.?|suite|unit|rm\.?|room|fl\.?|floor|"
    r"bldg\.?|building|dept\.?|trlr\.?|lot|no\.?|#)\s*"
    r"([a-z0-9\-]+)\s*$",
    re.I,
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def is_missing(value) -> bool:
    """True for None/NaN/empty/whitespace and the usual 'no data' sentinels.

    Datasets encode missingness as empty strings, the literal 'N/A', or real
    NaN depending on the source and the CSV parser. All of them must collapse
    to the same thing or the missingness features lie.
    """
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if value is pd.NaT:
        return True
    text = str(value).strip().lower()
    return text in _MISSING_TOKENS


def strip_accents(text: str) -> str:
    """Fold accented characters to ASCII ('Cafe' == 'Café')."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def squash(text: str) -> str:
    """Lowercase, de-accent, non-alphanumeric -> single space, trimmed."""
    if text is None:
        return ""
    text = strip_accents(str(text)).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _strip_trailing(tokens: List[str], vocabulary) -> List[str]:
    """Drop trailing tokens that are members of ``vocabulary`` (a set).

    Only the *end* of the string is considered, so a token that merely happens
    to be a suffix word in the middle ("Roadway Cafe") is left alone.
    """
    out = list(tokens)
    while out and out[-1] in vocabulary:
        out.pop()
    return out


def _map_trailing(tokens: List[str], vocabulary: Dict[str, str]) -> List[str]:
    """Rewrite trailing tokens through a canonicalization map.

    Used for street suffixes so "St" and "Street" both become "street". Only the
    trailing run is rewritten, and the loop continues while the *rewritten* form
    is still a key, so "Ave" -> "avenue" terminates in one step.
    """
    out = list(tokens)
    while out and out[-1] in vocabulary:
        out[-1] = vocabulary[out[-1]]
    return out


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def normalize_name(raw) -> Tuple[str, str]:
    """Return ``(name_norm, name_core)`` for a raw business name.

    ``name_norm``
        squashed text with punctuation removed and DBA markers resolved, legal
        suffix *kept*. This preserves the full identity signal.

    ``name_core``
        ``name_norm`` with trailing legal suffixes and generic business words
        additionally stripped. This is the string that lets
        "Acme Bicycle Company" and "Acme Bike Shop" meet in the middle.
    """
    if is_missing(raw):
        return "", ""

    text = strip_accents(str(raw)).lower()
    # Resolve dba/aka: everything before the marker is the primary name.
    # "Acme LLC dba Acme Shop" -> "Acme LLC".
    dba_match = re.search(
        r"\b(?:" + "|".join(sorted(DBA_MARKERS, key=len, reverse=True)) + r")\b", text
    )
    if dba_match:
        text = text[: dba_match.start()]

    tokens = [t for t in squash(text).split() if t]
    if not tokens:
        return "", ""

    name_norm = " ".join(tokens)
    # Drop trailing legal suffixes, then trailing generic words
    # ("Acme Bicycle Company LLC" -> "Acme Bicycle").
    core_tokens = _strip_trailing(tokens, LEGAL_SUFFIXES)
    core_tokens = _strip_trailing(core_tokens, GENERIC_BUSINESS_WORDS)
    # A leading "the" is noise in every source.
    if core_tokens and core_tokens[0] == "the":
        core_tokens = core_tokens[1:]
    # If stripping ate the whole name ("The Shop"), fall back to name_norm --
    # a short name is still far better than nothing.
    name_core = " ".join(core_tokens) if core_tokens else name_norm
    return name_norm, name_core


def name_tokens(name_norm: str) -> List[str]:
    return name_norm.split() if name_norm else []


# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------

def _is_address_tail(tokens: List[str], state: Optional[str]) -> bool:
    """True if this token list is addressing boilerplate rather than a city.

    Peels, from the right: a zip (and its +4 extension, which squashing already
    split into a separate token), a state token, and a secondary-unit
    designator. The chunk is boilerplate if nothing is left over.

    Deliberate limitation: a *multi-word* state name ("North Carolina") is NOT
    treated as a tail, because "New York", "Virginia", "Georgia", "Delaware" and
    friends are also city names and only position disambiguates them. Guessing
    "state" and swallowing the city destroys a high-value match feature,
    whereas guessing "city" and returning "north carolina" costs one weak
    feature. Where a dedicated city column exists it always wins anyway.
    """
    toks = [t for t in tokens if t]

    while toks and toks[-1].isdigit() and len(toks[-1]) in (4, 5):
        toks.pop()

    if toks:
        t = toks[-1]
        # Case-insensitive: tokens are lowercased, the state vocabulary is upper.
        if t.upper() in _VALID_STATES and (state is None or t.upper() == state):
            toks.pop()
        elif t in STATE_ABBREV and (state is None or STATE_ABBREV[t] == state):
            toks.pop()

    if len(toks) >= 2 and toks[-2] in UNIT_DESIGNATORS:
        del toks[-2:]

    return not toks


def parse_address(
    raw, raw_city=None, raw_state=None, raw_zip=None
) -> Dict[str, Optional[str]]:
    """Split a free-text address into comparable components.

    Handles the three shapes we actually see:

    1. fully comma-delimited  "123 N Main St, Ste 200, Springfield, IL 62704"
    2. trailing state+zip     "123 N Main St Ste 200 Springfield IL 62704"
    3. zip only               "PO Box 1234, Chicago, 60601"

    Dedicated columns win when present; otherwise city/state/zip are recovered
    from the address text. Returns a dict with keys:
    ``house_number, street_dir, street_name, street_suffix, unit, po_box,
      city, state, zip5, address_norm``.
    """
    out: Dict[str, Optional[str]] = {
        "house_number": None,
        "street_dir": None,
        "street_name": None,
        "street_suffix": None,
        "unit": None,
        "po_box": None,
        "city": None,
        "state": None,
        "zip5": None,
        "address_norm": None,
    }

    raw_text = "" if is_missing(raw) else strip_accents(str(raw)).lower()
    tokens = [t for t in squash(raw_text).split() if t]

    # ---- PO box -----------------------------------------------------------
    # A PO box is the whole "street" for that record, so capture it and delete
    # the phrase from the working text -- otherwise "box" and the box number get
    # parsed as a street name and the city ends up wrong.
    po = _POBOX_RE.search(raw_text)
    if po:
        out["po_box"] = squash(po.group(1))
        # Match the dotted forms too, otherwise "P.O. Box 99" leaves "p o" behind.
        raw_text = re.sub(r"\bp\.?\s*o\.?\s*box\s+\w+", " ", raw_text, flags=re.I)
        tokens = [t for t in squash(raw_text).split() if t]

    # ---- zip --------------------------------------------------------------
    zip5 = None
    zip_source = raw_zip
    if is_missing(zip_source) and raw_text:
        m = _ZIP_RE.search(raw_text)
        if m:
            zip5 = m.group(1)
    if not is_missing(zip_source):
        digits = _NON_DIGIT_RE.sub("", str(zip_source))
        if len(digits) >= 5:
            zip5 = digits[:5].zfill(5)
    out["zip5"] = zip5

    # ---- state ------------------------------------------------------------
    state = None
    if not is_missing(raw_state):
        state = normalize_state(raw_state)
    if state is None and raw_text:
        # Accept a trailing 2-letter postal code OR a spelled-out state name
        # ("... Springfield, Illinois 62704"), including two-word names such as
        # "North Carolina". Only the tail is considered so a street named
        # "North Ave" is never mistaken for a state.
        tail = tokens[-3:]
        for width in (2, 1):
            for start in range(0, max(0, len(tail) - width) + 1):
                candidate = " ".join(tail[start:start + width])
                if candidate in STATE_ABBREV:
                    state = STATE_ABBREV[candidate]
                    break
                if len(candidate) == 2 and candidate.upper() in _VALID_STATES:
                    state = candidate.upper()
                    break
            if state:
                break
    out["state"] = state

    # ---- split the street line off the front ------------------------------
    # Work right-to-left instead of trying to excise substrings from the raw
    # text. This is order-independent, so "Springfield IL 62704" is peeled off
    # as (state, zip) before the city is read, and multi-word cities such as
    # "Winston Salem" survive intact.
    tail_tokens = list(tokens)

    if zip5 and tail_tokens and _ZIP_RE.fullmatch(tail_tokens[-1] or ""):
        tail_tokens.pop()
    if state and tail_tokens:
        if tail_tokens[-1] == state.lower():
            tail_tokens.pop()
        elif tail_tokens[-1] in STATE_ABBREV and STATE_ABBREV[tail_tokens[-1]] == state:
            tail_tokens.pop()
    if not is_missing(raw_zip) and zip5 is None and tail_tokens and tail_tokens[-1].isdigit():
        tail_tokens.pop()  # zip we could not parse, but it is still a tail token

    # Remaining: "<street> <city>". Find the street suffix and split there.
    # When commas are present the first comma-delimited chunk is the street
    # line, which is more reliable than searching for the suffix.
    city = None
    st_tokens: List[str] = []
    rest_tokens: List[str] = []

    if out["po_box"]:
        # A PO box has no street line. What follows it is city/state/zip only,
        # so the first remaining chunk is the city, never a street name.
        chunks = [squash(c) for c in raw_text.split(",")]
        chunks = [c for c in chunks if c]
        if chunks:
            city = chunks[0]
        st_tokens, rest_tokens = [], []
    elif "," in raw_text:
        chunks = [squash(c) for c in raw_text.split(",")]
        chunks = [c for c in chunks if c]
        st_tokens = chunks[0].split() if chunks else []
        # Scan from the right for the first chunk that is *not* purely
        # geography/addressing boilerplate. That chunk is the city.
        #
        # Doing it this way rather than "peel the state off the last chunk"
        # matters for names that are both: "New York" is a city here but a state
        # in "North Carolina"-style tails, and only position disambiguates them.
        for chunk_tokens in reversed([c.split() for c in chunks[1:]]):
            if _is_address_tail(chunk_tokens, state):
                continue
            city = " ".join(chunk_tokens)
            break
        # A city that duplicates the street line is a parse artifact, not a city.
        if city and city == " ".join(st_tokens):
            city = None
    else:
        suffix_idx = None
        for i, tok in enumerate(tail_tokens):
            if tok in STREET_SUFFIXES:
                suffix_idx = i
                break
        if suffix_idx is not None:
            st_tokens = tail_tokens[: suffix_idx + 1]
            rest_tokens = tail_tokens[suffix_idx + 1:]
        elif tail_tokens and tail_tokens[0].isdigit() and len(tail_tokens) >= 3:
            # No recognizable suffix: assume a two-token street line.
            st_tokens = tail_tokens[:2]
            rest_tokens = tail_tokens[2:]
        else:
            st_tokens = list(tail_tokens)
            rest_tokens = []

    # ---- secondary unit ----------------------------------------------------
    # Peel the unit BEFORE deriving the city, because in the no-comma form the
    # unit sits in `rest_tokens` ahead of the city and would otherwise be
    # swallowed by it ("123 Main Street Suite 200" -> city "suite 200").
    unit = None
    if rest_tokens and rest_tokens[0] in UNIT_DESIGNATORS and len(rest_tokens) >= 2:
        unit = squash(rest_tokens[1])
        del rest_tokens[:2]
    else:
        # Some sources run the unit on inside the street chunk ("123 N Main St
        # Ste 200, ..."). Cut at the LAST designator so the house number and
        # street name survive.
        for j in range(len(st_tokens) - 1, 0, -1):
            if st_tokens[j] in UNIT_DESIGNATORS and j + 1 < len(st_tokens):
                unit = squash(st_tokens[j + 1])
                del st_tokens[j:]
                break
    out["unit"] = unit

    if city is None and rest_tokens:
        city = " ".join(rest_tokens) or None

    # Dedicated city column always wins over the text-derived guess.
    if not is_missing(raw_city):
        city = squash(raw_city)
    out["city"] = city

    if not st_tokens:
        out["address_norm"] = squash(raw_text) or None
        return out

    # ---- house number -----------------------------------------------------
    hm = _HOUSE_NUM_RE.match(" ".join(st_tokens))
    if hm:
        out["house_number"] = squash(hm.group(1))
        st_tokens = hm.group(2).split()

    # ---- directional prefix ----------------------------------------------
    if st_tokens and st_tokens[0] in DIRECTIONS:
        out["street_dir"] = DIRECTIONS[st_tokens[0]]
        st_tokens = st_tokens[1:]

    # ---- suffix (canonicalized) ------------------------------------------
    if st_tokens and st_tokens[-1] in STREET_SUFFIXES:
        out["street_suffix"] = STREET_SUFFIXES[st_tokens[-1]]
        st_tokens = st_tokens[:-1]

    out["street_name"] = " ".join(st_tokens) if st_tokens else None
    out["address_norm"] = squash(raw_text) or None
    return out


# ---------------------------------------------------------------------------
# Phone / zip / state / category
# ---------------------------------------------------------------------------

def normalize_phone(raw) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(phone10, phone7)``.

    ``phone10`` is the national 10-digit number; ``phone7`` is the local
    exchange+line. Source 3 frequently publishes only the 7-digit local number,
    so we keep both and let the feature layer compare 10-digit first and fall
    back to 7-digit.

    Country code: an 11-digit number starting with 1 is assumed NANP. Anything
    else is returned as-is digit-stripped when it is long enough to be a real
    number, and dropped otherwise (avoids matching on "1" or "12").
    """
    if is_missing(raw):
        return None, None
    text = strip_accents(str(raw)).lower()
    # Drop a trailing extension: "(555) 123-4567 x89" -> extension is noise.
    text = re.sub(r"(?:x|ext\.?|extension)\s*\d+\s*$", "", text)
    digits = _NON_DIGIT_RE.sub("", text)
    if not digits:
        return None, None
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return digits, digits[-7:]
    if len(digits) == 7:
        # Local number with no area code -- comparable via phone7 only.
        return None, digits
    return None, None


def normalize_zip(raw) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(zip5, zip3)``. zip3 is the section-center prefix."""
    if is_missing(raw):
        return None, None
    digits = _NON_DIGIT_RE.sub("", str(raw))
    if len(digits) < 5:
        return None, None
    zip5 = digits[:5]
    return zip5, zip5[:3]


def normalize_state(raw) -> Optional[str]:
    """Full state name or postal abbreviation -> 2-letter code."""
    if is_missing(raw):
        return None
    text = squash(raw)
    if not text:
        return None
    if text in STATE_ABBREV:
        return STATE_ABBREV[text]
    upper = text.upper()
    if len(upper) == 2 and upper in set(STATE_ABBREV.values()):
        return upper
    return None


def _singularize(word: str) -> str:
    """Very small, deliberately conservative plural stripper.

    Over-eager singularization hurts more than it helps here ("Glass" ->
    "Glas" is wrong), so we only handle the unambiguous regular cases.
    """
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_category(raw) -> str:
    """Map a raw industry label onto the shared vocabulary in CATEGORY_SYNONYMS."""
    if is_missing(raw):
        return ""
    text = squash(raw)
    if not text:
        return ""
    if text in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[text]
    singular = " ".join(_singularize(t) for t in text.split())
    if singular in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[singular]
    # Unknown label: singularized text stands in for itself so that two
    # sources using the same unrecognized label still agree exactly.
    return singular


def normalize_latlon(lat_raw, lon_raw) -> Tuple[float, float]:
    """Return ``(lat, lon)`` as floats, NaN when absent or out of range."""
    if is_missing(lat_raw) or is_missing(lon_raw):
        return np.nan, np.nan
    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except (TypeError, ValueError):
        return np.nan, np.nan
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return np.nan, np.nan
    return lat, lon


# ---------------------------------------------------------------------------
# Frame-level driver
# ---------------------------------------------------------------------------

def clean_frame(df: pd.DataFrame, field_map: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Normalize a raw source frame into canonical columns.

    Adds, in addition to the normalized values:
      * ``<field>_missing`` booleans for every key field
      * ``n_missing`` and ``missing_ratio`` summary counters
    """
    out = pd.DataFrame(index=df.index)

    def raw_for(canonical: str):
        col = field_map.get(canonical)
        if col is None or col not in df.columns:
            # A uniform NaN column: the source simply does not carry this field.
            return pd.Series(np.nan, index=df.index, dtype=object)
        return df[col]

    # ---- name -------------------------------------------------------------
    name_raw = raw_for("name")
    names = [normalize_name(v) for v in name_raw]
    out["name_norm"] = [n for n, _ in names]
    out["name_core"] = [c for _, c in names]
    out["name_tokens"] = out["name_norm"].map(name_tokens)

    # ---- address ----------------------------------------------------------
    parsed = [
        parse_address(a, c, s, z)
        for a, c, s, z in zip(
            raw_for("address"), raw_for("city"),
            raw_for("state"), raw_for("zip"),
        )
    ]
    for key in ("house_number", "street_dir", "street_name", "street_suffix",
                "unit", "po_box", "city", "state", "zip5", "address_norm"):
        out[key if key not in ("city", "state") else key + "_norm"] = [p[key] for p in parsed]
    # `city`/`state` become city_norm/state_norm for symmetry.
    out["city_norm"] = out.pop("city_norm")
    out["state_norm"] = out.pop("state_norm")

    # Dedicated zip column wins over whatever we scraped from the address.
    zips = [normalize_zip(v) for v in raw_for("zip")]
    out["zip5"] = [z[0] if z[0] else p["zip5"] for z, p in zip(zips, parsed)]
    out["zip3"] = [z[1] if z[1] else (p["zip5"][:3] if p["zip5"] else None)
                   for z, p in zip(zips, parsed)]

    # ---- phone ------------------------------------------------------------
    phones = [normalize_phone(v) for v in raw_for("phone")]
    out["phone10"] = [p[0] for p in phones]
    out["phone7"] = [p[1] for p in phones]

    # ---- category ---------------------------------------------------------
    out["category_norm"] = [normalize_category(v) for v in raw_for("category")]

    # ---- geo --------------------------------------------------------------
    lats, lons = [], []
    for la, lo in zip(raw_for("lat"), raw_for("lon")):
        lat, lon = normalize_latlon(la, lo)
        lats.append(lat)
        lons.append(lon)
    out["lat"] = lats
    out["lon"] = lons

    # ---- explicit missingness indicators ----------------------------------
    # `address` missingness is derived from address_norm so that a record whose
    # address only exists as free text still counts as present.
    derived_source = {
        "name": out["name_norm"],
        "address": out["address_norm"],
        "city": out["city_norm"],
        "state": out["state_norm"],
        "zip": out["zip5"],
        "phone": out["phone10"].fillna(out["phone7"]),
        "category": out["category_norm"],
    }
    for field, series in derived_source.items():
        out[field + "_missing"] = series.map(is_missing).astype(bool)
    out["n_missing"] = sum(out[f + "_missing"].astype(int) for f in KEY_FIELDS)
    out["missing_ratio"] = out["n_missing"] / float(len(KEY_FIELDS))

    return out


# ---------------------------------------------------------------------------
# Minimal real-schema cleaning (entity_id / business_name / business_address /
# country) -- the ML Challenge layout.
#
# The sources share no columns beyond those four, and `business_address` is one
# opaque free-text field rather than a pre-split set. Everything the pipeline
# needs beyond the raw four therefore has to be *derived* here, once, so the
# blocking keys, the pair features and the review output can never disagree
# about what "the city" of a record is.
#
# `clean_frame` above is kept untouched for the richer benchmark schema (it
# depends on phone/category/geo columns that do not exist here).
# ---------------------------------------------------------------------------

# Values that look like data but carry no identity. `is_missing` already covers
# the empty/NaN/'N/A' family; these are the *semantic* placeholders that survive
# as real strings. A record called "Unknown" must never become a blocking key or
# a name-similarity contributor, or thousands of rows collapse into one block.
PLACEHOLDER_TOKENS = {
    "na", "n a", "n/a", "n.a.", "nil", "none", "null", "nan", "unknown",
    "not known", "not available", "not applicable", "no data", "no name",
    "unnamed", "untitled", "unspecified", "missing", "empty", "-", "--", "---",
    "?", ".", "0", "00", "000", "test", "sample", "placeholder", "xxx", "tbd",
}

# Canonical field names the minimal cleaner understands. Anything else in a
# source's `fields` map is ignored, so an accidental extra mapping is harmless.
MINIMAL_FIELDS = ("name", "address", "country")


def is_placeholder(value) -> bool:
    """True for "N/A"-like non-answers, including ``Unnamed: 0``-style labels.

    Distinct from `is_missing`, which is about *absence*. A placeholder is
    present text that asserts nothing, and the two need separate flags because
    they imply different things to the model: missing means the source did not
    publish the field, a placeholder means it published nothing useful.
    """
    if is_missing(value):
        return True
    text = squash(value)
    if text in PLACEHOLDER_TOKENS:
        return True
    # "unnamed 0", "unknown 2" -- pandas' default header for a blank column.
    return text.startswith("unnamed") or text.startswith("unknown name")


def safe_value(value) -> str:
    """A blocking-safe string: '' for missing *and* for placeholders.

    Never returns a value that could collide across unrelated records, which is
    what keeps an empty field from becoming a 2-million-row block.
    """
    if is_placeholder(value):
        return ""
    return squash(value)


def extract_city_state(raw_address) -> Tuple[Optional[str], Optional[str]]:
    """Extract ``(city, state)`` from ONE free-text ``business_address`` field.

    Single entry point for both the Stage 1 blocking key and the Stage 3 pair
    features, on purpose: if blocking extracted the city one way and the feature
    layer another, a pair could be blocked on a city the features then report as
    conflicting. Returns ``(None, None)`` when the address is missing, is a
    placeholder, or yields nothing usable.

    Two deliberate differences from `parse_address`, which is tuned for the
    richer benchmark schema and for US-style addresses:

    * **A free-text address field has no single well-defined city.** "BROWNING,
      19 VAC RD, MT" puts the town first and the street second; "Unit APT 1,
      Anchorage, AK, 1350 27th Avenue" puts the unit first. So the candidates
      are collected as a *set* (see `locality_candidates`) and the returned
      city is the most plausible member of it, rather than the first chunk the
      parser stumbles on.
    * **A state is only a state if it is the last thing in the address** (after
      peeling a trailing postcode). Otherwise "Draper City (sl Co), UT" yields
      the state "CO", scraped out of a parenthetical, and a correct match gets
      vetoed for a state conflict that does not exist.

    Use `locality_candidates` directly when the whole set matters -- it is what
    the pair features compare.
    """
    if is_placeholder(raw_address):
        return None, None
    candidates = locality_candidates(raw_address)
    if not candidates:
        return None, None
    return _primary_locality(candidates), _trailing_region(raw_address)


def locality_candidates(raw_address) -> List[str]:
    """Every plausible locality string in one free-text address.

    For "Miryalguda Mandal, Survey No. 328/A/1, Miryalguda, Telangana" this
    returns ``["yadgarpally village", "miryalguda mandal", "miryalguda"]`` and
    drops the survey number and the region. Comparing two addresses is then a
    set-intersection question -- "miryalguda mandal" vs "miryalguda" is a hit --
    which is a far more honest test than equality on one arbitrarily chosen
    token.
    """
    if is_placeholder(raw_address):
        return []
    text = str(raw_address).lower()
    # A PO box is not a locality, and "box 42" must not become one.
    text = re.sub(r"\bp\.?\s*o\.?\s*box\s+\w+", " ", text)

    out: List[str] = []
    chunks = [c for c in text.split(",") if c.strip()]
    last = len(chunks) - 1
    for i, chunk in enumerate(chunks):
        tokens = [t for t in squash(chunk).split() if t]
        # Peel a trailing postcode from any chunk: it is geography, not a city.
        while tokens and _POSTCODE_TOKEN_RE.fullmatch(tokens[-1]):
            tokens.pop()
        # Peel a trailing region only from the LAST chunk. A region belongs at
        # the end of an address, and peeling it anywhere else eats innocent
        # words: "Draper City (sl Co), UT" would otherwise lose "Co"
        # (Colorado) out of the middle of a company name.
        if i == last:
            tokens = _peel_region(tokens)
        if not tokens:
            continue
        candidate = " ".join(tokens)
        if _is_locality_candidate(tokens):
            out.append(candidate)
    return out


def _primary_locality(candidates: Sequence[str]) -> Optional[str]:
    """The single best locality string out of the candidate set.

    "Best" is the *latest* usable candidate: in a free-text address the locality
    trails the street it belongs to, so "Bangalore South" beats "Karnataka" and
    "Bengaluru" beats a leading "Near SBI ATM" landmark. Length breaks ties
    between two equally-late candidates.
    """
    usable = [
        (i, c) for i, c in enumerate(candidates)
        if not _is_region(c) and not _looks_like_street(c.split())
    ]
    if not usable:
        # Everything looked like a street or a region: take the longest
        # candidate anyway, because a wrong city is a weak feature whereas no
        # city at all switches the city comparison off for the pair.
        return max(candidates, key=lambda c: len(c.split())) if candidates else None
    return max(usable, key=lambda ic: (ic[0], len(ic[1].split())))[1]


def _is_locality_candidate(tokens: List[str]) -> bool:
    """Reject address chunks that are plainly not a place name.

    Unit designators ("ste 200", "2nd floor"), survey/plot ids, landmark
    references ("near sbi atm") and street lines all have to go, or one of them
    becomes "the city" and every real match in the actual city looks like a
    conflict.
    """
    if not tokens:
        return False
    if any(t in _NON_LOCALITY_TOKENS for t in tokens):
        return False
    if tokens[0] in _LANDMARK_PREFIXES:
        return False
    if _FLAT_ID_RE.match(" ".join(tokens)):
        return False
    if all(t.isdigit() for t in tokens):
        return False
    return True


def _looks_like_street(tokens: List[str]) -> bool:
    """True when a chunk reads as a street line rather than a locality."""
    if not tokens:
        return False
    if re.match(r"^\d+[a-z]?$", tokens[0]) and len(tokens) <= 5:
        return True
    if tokens[-1] in STREET_SUFFIXES:
        return True
    return False


def _is_region(text: str) -> bool:
    """True when the whole string is an administrative region name/code."""
    squashed = squash(text)
    if not squashed:
        return False
    if squashed in REGION_ABBREV:
        return True
    return len(squashed) <= 3 and squashed.upper() in _REGION_CODES


def _peel_region(tokens: List[str]) -> List[str]:
    """Drop a trailing region token, including two-word region names."""
    out = list(tokens)
    while out:
        for width in (2, 1):
            if len(out) < width:
                continue
            candidate = " ".join(out[-width:])
            if _is_region(candidate):
                del out[-width:]
                break
        else:
            break
    return out


def _trailing_region(raw_address) -> Optional[str]:
    """The canonical region code, but only if the address *ends* with it.

    "123 N Main St, Springfield, IL 62704" -> IL. "Draper City (sl Co), UT" ->
    UT. "1350 27th Avenue" -> None. The trailing-position requirement is what
    keeps a region name from being scraped out of the middle of a free-text
    address, which was the single largest source of phantom state conflicts:
    the older parser read "Draper City (sl Co), UT" as state **CO**.
    """
    if is_placeholder(raw_address):
        return None
    tokens = [t for t in squash(raw_address).split() if t]
    # A postcode can sit between the region and the end ("IL 62704").
    while tokens and _POSTCODE_TOKEN_RE.fullmatch(tokens[-1]):
        tokens.pop()
    if not tokens:
        return None
    return _region_code(tokens[-1])


def _region_code(text: str) -> Optional[str]:
    """Full region name or code -> canonical code, else None."""
    squashed = squash(text)
    if not squashed:
        return None
    if squashed in REGION_ABBREV:
        return REGION_ABBREV[squashed]
    upper = squashed.upper()
    if len(upper) <= 3 and upper in _REGION_CODES:
        return upper
    return None



def clean_entity_frame(
    df: pd.DataFrame,
    field_map: Dict[str, Optional[str]],
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a real-schema source frame into the columns the pipeline uses.

    ``field_map`` maps the canonical names ``name`` / ``address`` / ``country``
    onto the source's actual column names (``None`` when the source lacks the
    field), so every path and column name stays configurable.

    Emits, beyond the normalized text:
      * ``name_norm`` / ``name_core`` / ``name_tokens`` -- name, legal suffixes
        stripped in ``name_core`` (the form used for blocking keys and for
        phonetics).
      * ``address_norm`` / ``city_norm`` / ``state_norm`` -- the single address
        blob plus the two components extracted from it.
      * ``country_norm`` -- squashed country, empty when missing/placeholder.
      * ``*_missing`` booleans and the two semantic flags
        ``address_parse_failed`` and ``name_placeholder``.
    """
    out = pd.DataFrame(index=df.index)

    def raw_for(canonical: str) -> pd.Series:
        col = field_map.get(canonical)
        if col is None or col not in df.columns:
            # The source does not carry this field at all: a uniform NaN column
            # so downstream code needs no None checks.
            return pd.Series(np.nan, index=df.index, dtype=object)
        return df[col]

    # ---- name ---------------------------------------------------------------
    name_raw = raw_for("name")
    names = [normalize_name(v) for v in name_raw]
    out["name_norm"] = [n for n, _ in names]
    out["name_core"] = [c for _, c in names]
    out["name_tokens"] = [name_tokens(n) for n in out["name_norm"]]
    # A name made entirely of suffix words ("The LLC") normalizes to nothing
    # useful; flag it so the feature layer can say "no name evidence" instead
    # of comparing two empty strings.
    out["name_placeholder"] = [is_placeholder(v) or not c for v, c in zip(name_raw, out["name_core"])]
    out["name_missing"] = out["name_norm"].map(lambda v: not bool(v))

    # ---- address ------------------------------------------------------------
    addr_raw = raw_for("address")
    parsed = [parse_address(a) for a in addr_raw]
    localities = [locality_candidates(a) for a in addr_raw]
    out["address_norm"] = [p["address_norm"] or "" for p in parsed]
    # `city_norm` is the single best guess: the blocking key, and what a
    # reviewer sees. `locality_tokens` is the full candidate set, which is what
    # the pair features compare -- see `locality_candidates` for why one token
    # is not enough for a free-text address field.
    out["city_norm"] = [(_primary_locality(c) or "") for c in localities]
    out["locality_tokens"] = localities
    out["state_norm"] = [_trailing_region(a) or "" for a in addr_raw]
    out["address_missing"] = [is_placeholder(a) or not p["address_norm"]
                              for a, p in zip(addr_raw, parsed)]
    # "Present but nothing could be read out of it" -- a different fact from
    # absent, and the model needs to tell them apart.
    out["address_parse_failed"] = [
        bool(not is_placeholder(a) and p["address_norm"]
             and not c and not p["house_number"] and not p["po_box"])
        for a, p, c in zip(addr_raw, parsed, localities)
    ]
    out["city_missing"] = [not bool(c) for c in out["city_norm"]]
    out["state_missing"] = [not bool(s) for s in out["state_norm"]]

    # ---- country ------------------------------------------------------------
    country_raw = raw_for("country")
    out["country_norm"] = [safe_value(v) for v in country_raw]
    out["country_missing"] = [not bool(c) for c in out["country_norm"]]

    # ---- original text, for the review and output files ---------------------
    # Never a model feature -- every one of these is in
    # `features.NON_FEATURE_COLUMNS` -- but carried so a reviewer sees the
    # record a decision was actually made about, not a bare probability.
    for _canonical in MINIMAL_FIELDS:
        _col = field_map.get(_canonical)
        out[f"{_canonical}_raw"] = (
            df[_col].fillna("").astype(str).to_numpy()
            if _col and _col in df.columns else np.full(len(df), "", dtype=object)
        )

    if id_col and id_col in df.columns:
        out[id_col] = df[id_col].to_numpy()
    return out
