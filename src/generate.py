"""Synthetic data generator.

Produces three CSVs that mimic the described feeds *and* a hidden truth file
listing the true matches. The truth file exists so the pipeline can report
blocking recall and so the end-to-end numbers are verifiable. It is never fed
to the model.

Why generate rather than just documenting the schema
----------------------------------------------------
Normalization and blocking decisions are only trustworthy if you can measure
them. "Did my blocking keys actually recall the known matches?" and "is the
threshold sane?" are unanswerable without labels, and manual labeling does not
scale to a smoke test. The generator gives a reproducible benchmark while the
real labeling workflow (src/labels.py) builds the set you will actually ship.

Each source gets a distinct corruption profile, because the hard part of ER is
that sources disagree in *systematic* ways, not random ones:

  Source 1 (reference)  clean, canonical, occasionally missing a field
  Source 2 (registry)   legal names with LLC/Inc, "Ste 200", office switchboard
                        phone (often a different area code), no coordinates
  Source 3 (directory)  marketing names, USPS-abbreviated streets, frequently
                        missing city, 7-digit phone, different category
                        vocabulary, geocoded but jittered

Plus deliberate traps, because a generator that only produces easy positives
flatters every downstream number:

  * chains        -- one Source 1 entity legitimately matching 2-3 records
  * distractors   -- near-identical names at a *different* address
  * moved         -- same entity, different address (a real relocation)
  * typos         -- character drop/swap/transpose inside the name
  * synonym names -- "Bicycle" vs "Bike", "Automotive" vs "Auto"
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

BRANDS = [
    "Acme", "Silver Lake", "Northstar", "Blue Ridge", "Cedar Grove", "Ironwood",
    "Maple Hill", "Riverbend", "Sunset", "Lakeside", "Pinecrest", "Fairview",
    "Grandview", "Harborview", "Kensington", "Willowbrook", "Stonegate",
    "Brookfield", "Highland", "Meridian", "Copperfield", "Elmhurst", "Foxglove",
    "Windward", "Thistledown", "Bramblewood", "Quarry Hill", "Amberly",
    "Clearwater", "Deepford", "Eastgate", "Fernwood", "Glenmore", "Havenwood",
]

# (display word used in the business name, canonical category). The category
# values must be canonical values from normalize.CATEGORY_SYNONYMS -- that dict
# is the single source of truth and also supplies Source 3's vocabulary below.
BUSINESS_TYPES = [
    ("Bicycle", "bike"), ("Automotive", "auto"), ("Restaurant", "restaurant"),
    ("Coffee", "coffee"), ("Bakery", "bakery"), ("Hardware", "hardware"),
    ("Plumbing", "plumbing"), ("Electrical", "electrical"),
    ("Landscaping", "landscaping"), ("Dental", "dental"), ("Legal", "legal"),
    ("Accounting", "accounting"), ("Insurance", "insurance"),
    ("Pet", "pet"), ("Floral", "floral"), ("Furniture", "furniture"),
    ("Gym", "fitness"), ("Barber", "beauty"), ("Cleaners", "cleaning"),
    ("Printing", "printing"), ("Pharmacy", "pharmacy"),
    ("Optometry", "optometry"), ("Childcare", "childcare"),
    ("Salon", "beauty"), ("Moving", "moving"), ("Storage", "storage"),
    ("Electric", "electric"), ("Roofing", "roofing"), ("Siding", "siding"),
    ("Pools", "pool"),
]

# Business-type word -> the marketing synonym Source 3 uses. These are real
# lexical substitutions, not abbreviations, which is why `name_core` alone is
# not enough and the model needs token-level fuzzy features.
MARKETING_SYNONYM = {
    "Bicycle": "Bike", "Automotive": "Auto", "Coffee": "Coffee",
    "Hardware": "Hardware", "Landscaping": "Landscaping", "Legal": "Legal",
    "Accounting": "Accounting", "Insurance": "Insurance", "Floral": "Floral",
    "Furniture": "Furniture", "Gym": "Fitness", "Barber": "Barber",
    "Cleaners": "Cleaners", "Printing": "Printing", "Pharmacy": "Pharmacy",
    "Optometry": "Optometry", "Childcare": "Childcare", "Salon": "Salon",
    "Moving": "Moving", "Storage": "Storage", "Electric": "Electrical",
    "Roofing": "Roofing", "Siding": "Siding", "Pools": "Pool",
    "Restaurant": "Restaurant", "Bakery": "Bakery", "Plumbing": "Plumbing",
    "Electrical": "Electrical", "Dental": "Dental", "Pet": "Pet",
}

STREET_NAMES = [
    "Main", "Oak", "Maple", "Cedar", "Elm", "Pine", "Washington", "Lake",
    "Highland", "Park", "Chestnut", "Walnut", "Lincoln", "Jefferson",
    "Willow", "Birch", "Spruce", "Cherry", "Poplar", "Hickory", "Sycamore",
    "Broadway", "Market", "Union", "Liberty", "Franklin", "Monroe",
]

STREET_TYPES = ["St", "Ave", "Rd", "Blvd", "Dr", "Ln", "Way", "Ct", "Pl"]

# (city, state, zip_prefix, lat, lon). Zip is built as prefix + 3 digits, which
# keeps zips inside one prefix geographically coherent -- important, because a
# zip-prefix blocking key would be useless if zips were random.
CITIES = [
    ("Springfield", "IL", "627", 39.7817, -89.6501),
    ("Chicago", "IL", "606", 41.8781, -87.6298),
    ("Aurora", "IL", "605", 41.7606, -88.3201),
    ("Naperville", "IL", "605", 41.7508, -88.1535),
    ("Columbus", "OH", "432", 39.9612, -82.9988),
    ("Cleveland", "OH", "441", 41.4993, -81.6944),
    ("Indianapolis", "IN", "462", 39.7684, -86.1581),
    ("Nashville", "TN", "372", 36.1627, -86.7816),
    ("Portland", "OR", "972", 45.5152, -122.6784),
    ("Eugene", "OR", "974", 44.0521, -123.0868),
    ("Phoenix", "AZ", "850", 33.4484, -112.0740),
    ("Tucson", "AZ", "857", 32.2226, -110.9747),
    ("Denver", "CO", "802", 39.7392, -104.9903),
    ("Boulder", "CO", "803", 40.0150, -105.2705),
    ("Austin", "TX", "787", 30.2672, -97.7431),
    ("Dallas", "TX", "752", 32.7767, -96.7970),
    ("Atlanta", "GA", "303", 33.7490, -84.3880),
    ("Savannah", "GA", "314", 32.0809, -81.0912),
    ("Raleigh", "NC", "276", 35.7796, -78.6382),
    ("Charlotte", "NC", "282", 35.2271, -80.8431),
    ("Philadelphia", "PA", "191", 39.9526, -75.1652),
    ("Pittsburgh", "PA", "152", 40.4406, -79.9959),
    ("Boston", "MA", "021", 42.3601, -71.0589),
    ("Worcester", "MA", "016", 42.2626, -71.8023),
    ("Minneapolis", "MN", "554", 44.9778, -93.2650),
    ("Rochester", "MN", "559", 44.0121, -92.4802),
    ("Kansas City", "MO", "641", 39.0997, -94.5786),
    ("New Orleans", "LA", "701", 29.9511, -90.0715),
    ("Seattle", "WA", "981", 47.6062, -122.3321),
    ("Spokane", "WA", "992", 47.6588, -117.4260),
]

LEGAL_FORMS = ["LLC", "Inc", "Corp", "Co", "Ltd", "LP"]

# Source 3 publishes a *different* controlled vocabulary from Sources 1/2, so
# the category feature has to survive a total string mismatch. Rather than
# maintaining a second hand-written table that can drift, derive Source 3's
# forms from the normalizer's crosswalk: for each canonical category, collect
# every surface form that maps to it and pick one that is not the plain form.
def _build_source3_vocab() -> Dict[str, List[str]]:
    from .normalize import CATEGORY_SYNONYMS

    by_canon: Dict[str, List[str]] = {}
    for surface, canon in CATEGORY_SYNONYMS.items():
        by_canon.setdefault(canon, []).append(surface)
    # The generator's own display words are the "plain" forms we want Source 3
    # to *differ* from.
    plain = {cat.lower() for _, cat in BUSINESS_TYPES}
    return {
        canon: [s for s in sorted(forms) if s not in plain and s != canon]
        for canon, forms in by_canon.items()
    }


CATEGORY_S3 = _build_source3_vocab()

USPS_ABBREV = {
    "Street": "St", "Avenue": "Ave", "Road": "Rd", "Boulevard": "Blvd",
    "Drive": "Dr", "Lane": "Ln", "Court": "Ct", "Place": "Pl",
    "Terrace": "Ter", "Circle": "Cir", "Parkway": "Pkwy", "Highway": "Hwy",
}


# ---------------------------------------------------------------------------
# Corruption helpers
# ---------------------------------------------------------------------------

class _IdMinter:
    """Mints guaranteed-unique record ids.

    A random suffix is not enough. An entity with two Source 2 records draws two
    independent 3-digit suffixes, and they collide with probability ~1/900 --
    which, across ~1600 rows, happens a few times per run. A duplicate id is not
    cosmetic: the final match table would list the same candidate id for two
    different records, and any merge on id would silently duplicate rows. The
    collision also inflates the labeled-pair count, because the labeling sheet
    keys on (s1_id, candidate_id, source).
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.used: set = set()

    def mint(self, entity_id: str, tag: str = "") -> str:
        while True:
            rid = f"{self.prefix}{entity_id[1:]}-{random.randrange(100, 1000)}{tag}"
            if rid not in self.used:
                self.used.add(rid)
                return rid


def _maybe(rng: random.Random, prob: float) -> bool:
    return rng.random() < prob


def _drop_random(value: str, rng: random.Random) -> str:
    """Simulate a typo: delete a random interior character."""
    if len(value) < 4:
        return value
    i = rng.randrange(1, len(value) - 1)
    return value[:i] + value[i + 1:]


def _transpose_adjacent(value: str, rng: random.Random) -> str:
    """Simulate a transposed pair -- the most common real-world typo."""
    if len(value) < 4:
        return value
    i = rng.randrange(1, len(value) - 2)
    return value[:i] + value[i + 1] + value[i] + value[i + 2:]


def _misspell(value: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return _drop_random(value, rng)
    if r < 0.8:
        return _transpose_adjacent(value, rng)
    return value + rng.choice("aeiou")


def _maybe_missing(value: str, rng: random.Random, prob: float) -> Optional[str]:
    """Return a sentinel the normalizer treats as missing, or the value."""
    if _maybe(rng, prob):
        return rng.choice(["", "N/A", "null", "-"])
    return value


def _jitter_geo(lat: float, lon: float, rng: random.Random, meters: float) -> Tuple[float, float]:
    """Offset a coordinate by up to ~`meters`."""
    dlat = rng.uniform(-meters, meters) / 111_320.0
    dlon = rng.uniform(-meters, meters) / (111_320.0 * max(0.2, abs(lat) / 90.0))
    return round(lat + dlat, 6), round(lon + dlon, 6)


# ---------------------------------------------------------------------------
# Entity construction
# ---------------------------------------------------------------------------

def _zip_pools(rng: random.Random, per_city: int = 6) -> Dict[str, List[str]]:
    """A small pool of zips per city.

    Real zips are not unique per business -- a city has a handful of dominant
    postcodes and dozens of businesses share each one. If every entity got a
    random 5-digit zip, ``zip5`` would be a near-perfect entity identifier and
    the model would score 95% importance on it while learning nothing about
    matching. Restricting each city to `per_city` zips reproduces the real
    collision rate and forces the name/address/phone features to do work.
    """
    pools: Dict[str, List[str]] = {}
    for city, _state, zip_prefix, _lat, _lon in CITIES:
        pools[city] = [
            f"{zip_prefix}{rng.randrange(0, 1000):03d}" for _ in range(per_city)
        ]
    return pools


def _build_entities(n: int, rng: random.Random, twin_rate: float = 0.08) -> List[Dict]:
    """Create n ground-truth businesses with their canonical field values.

    Also creates "twins": distinct real businesses that share a name and a
    postcode with an existing entity but sit at a different address. Twins are
    the hard negative this problem is really about. Without them every negative
    is trivially separable and the benchmark flatters any model -- in production
    the confusing candidates are the ones that *look* like matches.

    A twin is a legitimate separate entity, so it gets its own Source 1 record
    and its own true matches. The model is scored on whether it links each
    entity to its own records and not to its twin's.
    """
    entities: List[Dict] = []
    seen_names = set()
    zips = _zip_pools(rng)

    for idx in range(1, n + 1):
        # Retry until the canonical name is unique among *non-twins*. Near
        # duplicate names are what the model must survive, but two entities
        # identical in every field would be unmatchable noise rather than a
        # learnable hard case.
        for _ in range(50):
            brand = rng.choice(BRANDS)
            biz_type, _ = rng.choice(BUSINESS_TYPES)
            name = f"{brand} {biz_type}"
            if name not in seen_names:
                seen_names.add(name)
                break

        city, state, zip_prefix, lat, lon = rng.choice(CITIES)
        street_type = rng.choice(STREET_TYPES)
        entities.append({
            "entity_id": f"E{idx:06d}",
            "canonical_name": name,
            "legal_name": f"{name} {rng.choice(LEGAL_FORMS)}",
            "house_number": str(rng.randrange(1, 9800)),
            "street_name": rng.choice(STREET_NAMES),
            "street_type": street_type,
            "unit": str(rng.randrange(100, 999)) if _maybe(rng, 0.35) else None,
            "city": city,
            "state": state,
            "zip": rng.choice(zips[city]),
            "lat": lat,
            "lon": lon,
            # Source 1's phone is the storefront number; Source 2's registry
            # phone is often a corporate switchboard, i.e. a *different* real
            # number for the same entity. This is why phone is a feature and not
            # a join key.
            "phone_area": f"{rng.randrange(201, 989)}",
            "phone_exchange": f"{rng.randrange(200, 999)}",
            "phone_line": f"{rng.randrange(0, 10000):04d}",
            "category": dict(BUSINESS_TYPES)[name.split(" ", 1)[1]]
            if name.split(" ", 1)[1] in dict(BUSINESS_TYPES) else "restaurant",
            "biz_type": biz_type,
        })

    # ---- twins -------------------------------------------------------------
    n_twins = int(n * twin_rate)
    base_entities = list(entities)
    for k in range(n_twins):
        parent = rng.choice(base_entities)
        idx = n + k + 1
        # Same postcode, different address. Deliberately keep city/state/zip
        # identical so that only the street address and the phone separate them.
        # The twin shares the parent's zip exactly, which is the whole point:
        # zip agreement is no longer evidence of a match.
        # The twin is a few blocks away, not in another metro area.
        twin_lat, twin_lon = _jitter_geo(
            parent["lat"], parent["lon"], rng, meters=1800.0
        )
        twin = dict(parent)
        twin.update({
            "entity_id": f"E{idx:06d}",
            "legal_name": f"{parent['canonical_name']} {rng.choice(LEGAL_FORMS)}",
            "house_number": str(rng.randrange(1, 9800)),
            "unit": str(rng.randrange(100, 999)) if _maybe(rng, 0.35) else None,
            "lat": twin_lat,
            "lon": twin_lon,
            # A different real phone number: this is what tells the model that
            # "same name, same zip, different address" can be either a twin or a
            # relocation.
            "phone_area": f"{rng.randrange(201, 989)}",
            "phone_exchange": f"{rng.randrange(200, 999)}",
            "phone_line": f"{rng.randrange(0, 10000):04d}",
        })
        entities.append(twin)
    return entities


def _category_for(entity: Dict) -> str:
    return entity["category"]


# ---------------------------------------------------------------------------
# Per-source renderers
# ---------------------------------------------------------------------------

def _render_source1(entity: Dict, rng: random.Random) -> Dict:
    """Reference source: canonical, lightly perturbed."""
    name = entity["canonical_name"]
    if _maybe(rng, 0.12):
        name = entity["legal_name"]

    street = f"{entity['house_number']} {entity['street_name']} {entity['street_type']}"
    if entity["unit"]:
        street += f" Ste {entity['unit']}"
    full = f"{street}, {entity['city']}, {entity['state']} {entity['zip']}"
    lat, lon = entity["lat"], entity["lon"]

    return {
        "business_id": f"B{entity['entity_id'][1:]}",
        "business_name": _maybe_missing(name, rng, 0.02),
        "street_address": _maybe_missing(full, rng, 0.02),
        "city": _maybe_missing(entity["city"], rng, 0.04),
        "state": _maybe_missing(entity["state"], rng, 0.02),
        "zip_code": _maybe_missing(entity["zip"], rng, 0.03),
        "phone": _maybe_missing(
            f"({entity['phone_area']}) {entity['phone_exchange']}-{entity['phone_line']}",
            rng, 0.10),
        "category": _maybe_missing(_category_for(entity), rng, 0.05),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
    }


def _corrupt_address(entity: Dict, rng: random.Random) -> Tuple[str, str]:
    """Return (house_number, street_name) with a realistic keying error.

    Exact address agreement is the single easiest signal in this problem, and a
    benchmark where the address always matches teaches nothing. Real registries
    contain transposed house numbers and misspelled street names, so a fraction
    of records get one. The model then has to lean on name/phone/geo instead.
    """
    house = entity["house_number"]
    street = entity["street_name"]
    if _maybe(rng, 0.12) and len(house) >= 3:
        # Transpose the last two digits: 1234 -> 1243.
        house = house[:-2] + house[-1] + house[-2]
    elif _maybe(rng, 0.06):
        house = house + "A"          # "123" vs "123A"
    if _maybe(rng, 0.14):
        street = _misspell(street, rng)
    return house, street


def _render_source2(entity: Dict, rng: random.Random, ids: "_IdMinter") -> Dict:
    """Corporate registry: legal names, explicit units, switchboard phone, no geo."""
    name = entity["legal_name"]
    if _maybe(rng, 0.18):
        name = _misspell(name, rng)

    house, street_name = _corrupt_address(entity, rng)
    street_type = USPS_ABBREV.get(entity["street_type"], entity["street_type"])
    street = f"{house} {street_name} {street_type}"
    if entity["unit"]:
        street += f" Ste {entity['unit']}"
    full = f"{street}, {entity['city']}, {entity['state']} {entity['zip']}"

    # 30% of the time the registry lists the corporate switchboard rather than
    # the storefront line, so the phone legitimately disagrees.
    if _maybe(rng, 0.30):
        area = f"{rng.randrange(201, 989)}"
        exch = f"{rng.randrange(200, 999)}"
    else:
        area, exch = entity["phone_area"], entity["phone_exchange"]

    return {
        "registry_id": ids.mint(entity["entity_id"]),
        "entity_name": _maybe_missing(name, rng, 0.03),
        "addr_line1": _maybe_missing(full, rng, 0.03),
        "addr_city": _maybe_missing(entity["city"], rng, 0.05),
        "addr_state": _maybe_missing(entity["state"], rng, 0.02),
        "addr_zip": _maybe_missing(entity["zip"], rng, 0.04),
        "contact_phone": _maybe_missing(
            f"{area}-{exch}-{entity['phone_line']}", rng, 0.12),
        "naics_label": _maybe_missing(_category_for(entity), rng, 0.08),
        # Source 2 carries no coordinates at all.
        "lat": None,
        "lon": None,
    }


def _render_source3(entity: Dict, rng: random.Random, ids: "_IdMinter") -> Dict:
    """Local directory: marketing name, abbreviated streets, 7-digit phone, jittered geo."""
    brand = entity["canonical_name"].split(" ", 1)[0]
    # Marketing name: a real lexical synonym for the business-type word, and
    # frequently no category word at all. "Acme Bicycle" -> "Acme Bike".
    synonym = MARKETING_SYNONYM.get(entity["biz_type"], entity["biz_type"])
    if _maybe(rng, 0.55):
        name = f"{brand} {synonym}"
    else:
        name = entity["canonical_name"]
    if _maybe(rng, 0.20):
        name = _misspell(name, rng)
    if _maybe(rng, 0.10):
        name = f"{name} Shop"

    house, street_name = _corrupt_address(entity, rng)
    street_type = USPS_ABBREV.get(entity["street_type"], entity["street_type"])
    street = f"{house} {street_name} {street_type}"
    if entity["unit"] and _maybe(rng, 0.5):
        street += f" #{entity['unit']}"
    full = f"{street}, {entity['city']}, {entity['state']} {entity['zip']}"
    # A directory listing that never recorded the house number: the model has to
    # fall back to street + zip, where the number carries less information.
    if _maybe(rng, 0.08):
        full = f"{street_name} {street_type}, {entity['city']}, {entity['state']} {entity['zip']}"

    lat, lon = _jitter_geo(entity["lat"], entity["lon"], rng, meters=350.0)
    # A directory vocabulary that deliberately shares no characters with the
    # reference source's label -- this is what forces the category feature to
    # rely on the crosswalk rather than on string similarity.
    options = CATEGORY_S3.get(entity["category"]) or []
    cat = rng.choice(options) if options else entity["category"]

    # 45% of directory listings publish only the 7-digit local number.
    if _maybe(rng, 0.45):
        phone = f"{entity['phone_exchange']}-{entity['phone_line']}"
    else:
        phone = f"({entity['phone_area']}) {entity['phone_exchange']}-{entity['phone_line']}"

    return {
        "listing_id": ids.mint(entity["entity_id"]),
        "listing_name": _maybe_missing(name, rng, 0.05),
        "street": _maybe_missing(full, rng, 0.05),
        # City column is genuinely absent from this source (see schema.yaml).
        "state_code": _maybe_missing(entity["state"], rng, 0.04),
        "postal_code": _maybe_missing(entity["zip"], rng, 0.08),
        "phone_number": _maybe_missing(phone, rng, 0.20),
        "industry": _maybe_missing(cat, rng, 0.10),
        "geo_lat": lat,
        "geo_lon": lon,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate(
    n_entities: int = 1200,
    out_dir: str = "data/raw",
    truth_dir: str = "data/_truth",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Write source1/2/3 CSVs plus the hidden truth file. Returns the frames."""
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)

    entities = _build_entities(n_entities, rng)

    s1_rows: List[Dict] = []
    s2_rows: List[Dict] = []
    s3_rows: List[Dict] = []
    truth: List[Dict] = []
    s2_ids = _IdMinter("R")
    s3_ids = _IdMinter("L")

    for entity in entities:
        # ---- Source 1: exactly one record, it is the deduplicated reference.
        r1 = _render_source1(entity, rng)
        s1_rows.append(r1)
        s1_id = r1["business_id"]

        # ---- Source 2: 0, 1 or many records. Zero is common and must stay.
        for _ in range(rng.choices([0, 1, 2, 3], weights=[18, 52, 24, 6])[0]):
            r2 = _render_source2(entity, rng, s2_ids)
            # Relocation: a small share of records report a *different* address
            # for the same entity while keeping the same phone. This is the
            # deliberate contrast to a twin -- name matches, address disagrees,
            # and only the phone separates "moved" from "different business".
            # Without this case the model would learn "address disagrees =>
            # non-match", which silently loses every business that moved.
            if _maybe(rng, 0.05):
                r2["addr_line1"] = (
                    f"{rng.randrange(1, 9800)} {rng.choice(STREET_NAMES)} "
                    f"{rng.choice(STREET_TYPES)}, {entity['city']}, "
                    f"{entity['state']} {entity['zip']}"
                )
            s2_rows.append(r2)
            truth.append({"source1_id": s1_id, "candidate_id": r2["registry_id"],
                          "candidate_source": "source2", "label": 1})

        # ---- Source 3: 0, 1 or many records.
        for _ in range(rng.choices([0, 1, 2, 3], weights=[22, 50, 22, 6])[0]):
            r3 = _render_source3(entity, rng, s3_ids)
            if _maybe(rng, 0.05):
                r3["street"] = (
                    f"{rng.randrange(1, 9800)} {rng.choice(STREET_NAMES)} "
                    f"{rng.choice(STREET_TYPES)}, {entity['city']}, "
                    f"{entity['state']} {entity['zip']}"
                )
            s3_rows.append(r3)
            truth.append({"source1_id": s1_id, "candidate_id": r3["listing_id"],
                          "candidate_source": "source3", "label": 1})

    # ---- Traps -------------------------------------------------------------
    # Distractor A -- "moved" clone: a source-2 record re-registered in a
    # different city under a new id. The name matches almost exactly, so any
    # name-driven rule fires; only address/geo/phone disagree.
    for r2 in list(s2_rows):
        if _maybe(rng, 0.10):
            city, state, zip_prefix, _lat, _lon = rng.choice(CITIES)
            clone = dict(r2)
            clone["registry_id"] = r2["registry_id"] + "-D"
            clone["addr_city"] = city
            clone["addr_state"] = state
            clone["addr_zip"] = f"{zip_prefix}{rng.randrange(0, 1000):03d}"
            clone["addr_line1"] = (
                f"{rng.randrange(1, 9800)} {rng.choice(STREET_NAMES)} "
                f"{rng.choice(STREET_TYPES)}, {city}, {state} {clone['addr_zip']}"
            )
            s2_rows.append(clone)
            # Deliberately NOT appended to truth: this is a non-match.

    # Distractor B -- "unitmate": a *different* business in the same building.
    # This is the hard negative for address-based reasoning: city, state, zip,
    # street name and street suffix all agree, and the only evidence separating
    # it from a true match is the house number and the business name. Without
    # it a model can look great on the easy negatives and still be useless.
    for r3 in list(s3_rows):
        if _maybe(rng, 0.12):
            clone = dict(r3)
            clone["listing_id"] = r3["listing_id"] + "-U"
            # Same street line, different house number, different business.
            parts = r3["street"].split(" ", 1)
            clone["street"] = (
                f"{rng.randrange(1, 9800)} {parts[1]}" if len(parts) == 2 else r3["street"]
            )
            other_brand = rng.choice([b for b in BRANDS if b not in r3["listing_name"]])
            other_type = rng.choice(BUSINESS_TYPES)[0]
            clone["listing_name"] = f"{other_brand} {other_type}"
            other_cat = CATEGORY_S3.get(
                dict(BUSINESS_TYPES)[other_type], [other_type]
            )
            clone["industry"] = rng.choice(other_cat) if other_cat else other_type
            # Keep the city/zip identical on purpose: that is what makes it hard.
            s3_rows.append(clone)
            # Not in truth: a non-match.

    frames = {
        "source1": pd.DataFrame(s1_rows),
        "source2": pd.DataFrame(s2_rows),
        "source3": pd.DataFrame(s3_rows),
    }

    # Fail fast on duplicate ids. A duplicated id makes the final match table
    # ambiguous and inflates the labeled-pair count, and both failures are
    # silent -- far better to stop here than to debug it three stages later.
    id_cols = {
        "source1": "business_id",
        "source2": "registry_id",
        "source3": "listing_id",
    }
    for name, frame in frames.items():
        col = id_cols[name]
        n_unique = frame[col].nunique()
        if n_unique != len(frame):
            dupes = frame[col][frame[col].duplicated()].unique()[:5]
            raise AssertionError(
                f"{name}: {col} is not unique ({len(frame)} rows, "
                f"{n_unique} unique). Examples: {list(dupes)}"
            )

    for name, frame in frames.items():
        frame.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    truth_df = pd.DataFrame(truth)
    truth_df.to_csv(os.path.join(truth_dir, "truth_pairs.csv"), index=False)

    return frames
