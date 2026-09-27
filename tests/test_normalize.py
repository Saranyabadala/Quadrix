"""Regression tests for the normalizers.

These exist because normalization is the load-bearing step: if "123 N Main St"
and "123 North Main St." do not reduce to the same components, every string
similarity downstream is measuring abbreviations instead of identity, and the
model will happily learn the wrong thing.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.normalize import (  # noqa: E402
    normalize_category, normalize_name, normalize_phone, normalize_state,
    normalize_zip, parse_address,
)


def test_name_legal_suffix_and_generic_words():
    """The same entity under a legal name and a marketing name must share a core."""
    legal = normalize_name("The Acme Bicycle Co., LLC")
    registry = normalize_name("ACME BICYCLE COMPANY LLC")
    marketing = normalize_name("Acme Bike Shop")

    # The legal and registry forms must land on exactly the same core: the
    # whole point of stripping suffixes and generic words.
    assert legal[1] == "acme bicycle"
    assert registry[1] == "acme bicycle"

    # "Bike" vs "Bicycle" is a real synonym, not an abbreviation, so the cores
    # need not be equal -- that gap is what the fuzzy name features exist for.
    assert marketing[1] == "acme bike"

    # name_norm keeps the legal suffix; name_core drops it. Both representations
    # are needed downstream, so neither may be empty.
    assert "llc" in legal[0].split()
    assert "llc" not in legal[1].split()
    assert legal[0] and legal[1]


def test_name_dba_prefers_leading_name():
    norm, core = normalize_name("Cafe de Paris LLC dba Little Paris")
    assert core == "cafe de paris"
    assert "little" not in norm


def test_name_missing_tokens():
    for blank in (None, "", "N/A", "n/a", "null", "  ", "-"):
        assert normalize_name(blank) == ("", "")


def test_name_falls_back_when_suffix_stripping_empties_it():
    """'The Shop' must not normalize to an empty string."""
    norm, core = normalize_name("The Shop")
    assert norm == "the shop"
    assert core, "name_core must never be empty when name_norm is not"


def test_address_surfaces_of_one_entity_agree():
    """Five source-specific surface forms of the same address must agree."""
    forms = [
        ("123 North Main St., Ste. 200, Springfield, IL 62704", None, None, None),
        ("123 N MAIN ST STE 200, SPRINGFIELD, IL  62704-1234", None, None, None),
        ("123 N Main St", "Springfield", "IL", "62704"),
        ("123 N Main St, Springfield, Illinois 62704", None, None, None),
    ]
    parsed = [parse_address(*f) for f in forms]
    for p in parsed:
        assert p["house_number"] == "123", p
        assert p["street_name"] == "main", p
        assert p["street_suffix"] == "street", p
        assert p["city"] == "springfield", p
        assert p["state"] == "IL", p
        assert p["zip5"] == "62704", p
    # The unit is present in only two of the five, and must never be confused
    # with the house number.
    assert parsed[0]["unit"] is None
    assert parsed[1]["unit"] == "200"
    assert parsed[0]["house_number"] == "123"


def test_address_directional_and_multiword_city():
    p = parse_address("789 N Elm Rd Salt Lake City UT 84101")
    assert p["street_dir"] == "n"
    assert p["street_name"] == "elm"
    assert p["street_suffix"] == "road"
    assert p["city"] == "salt lake city"
    assert p["state"] == "UT"


def test_address_city_state_name_collision_prefers_city():
    """'New York' as a city must not be swallowed as a state."""
    p = parse_address("55 Broadway, New York, NY, 10006")
    assert p["city"] == "new york"
    assert p["state"] == "NY"


def test_address_po_box():
    p = parse_address("P.O. Box 99, Denver, CO 80202")
    assert p["po_box"] == "99"
    assert p["city"] == "denver"
    assert p["state"] == "CO"
    assert p["zip5"] == "80202"
    assert p["street_name"] is None


def test_address_full_street_word_is_a_recognized_suffix():
    p = parse_address("123 North Main Street Suite 200")
    assert p["street_suffix"] == "street"
    assert p["unit"] == "200"
    assert p["street_name"] == "main"


def test_address_missing():
    for blank in (None, "", "N/A"):
        p = parse_address(blank)
        assert p["house_number"] is None
        assert p["address_norm"] in (None, "")


def test_phone_variants_agree():
    """Every formatting of one number must reduce to the same 10 digits."""
    for variant in ["(217) 555-0142", "217-555-0142", "+1 217 555 0142",
                    "1.217.555.0142", "2175550142 x89"]:
        assert normalize_phone(variant)[0] == "2175550142", variant


def test_phone_seven_digit_and_unusable():
    assert normalize_phone("555-0142") == (None, "5550142")
    # Vanity numbers have no digits, so they cannot be normalized; dropping them
    # is correct, matching on letters would be worse.
    assert normalize_phone("1-800-FLOWERS") == (None, None)
    assert normalize_phone(None) == (None, None)


def test_zip():
    assert normalize_zip("62704-1234")[0] == "62704"
    assert normalize_zip(94105)[0] == "94105"
    assert normalize_zip("274") == (None, None)


def test_state():
    assert normalize_state("Illinois") == "IL"
    assert normalize_state("il") == "IL"
    assert normalize_state("IL") == "IL"
    assert normalize_state("Ontario") is None


def test_category_crosswalk():
    assert normalize_category("Restaurants") == "restaurant"
    assert normalize_category("Eating Places") == "restaurant"
    assert normalize_category("Auto Repair Shops") == "auto"
