"""Швидкий пошук збігів у сніпетах (лаба 8) мусить давати те саме, що й старий вираз."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.documents import strip_boilerplate
from findex.snippets import _find_literals, _find_matches, _find_with_regex

CASES = [
    (
        "The white Whale, the WHALE; whales and whaler. Seasick sea-dogs: sea! seas",
        {"whale", "sea"},
    ),
    ("sea seas season seaside", {"sea", "seas"}),  # один терм є початком іншого
    ("_sea_ sea_ _sea 9sea sea9 sea", {"sea"}),  # підкреслення й цифри це частини слова
    ("well-known well-knownx well well-", {"well", "well-known"}),  # дефіс: старий шлях
    ("don't dont Don’t", {"don't", "dont"}),  # апостроф: старий шлях
    ("Ünïcode Straße STRASSE strasse", {"strasse", "ünïcode"}),  # не ASCII-терми: старий шлях
    ("Kelvin sign: K and ſ and İ and ı", {"k", "s", "i"}),  # символи, що склеюються з ASCII
    ("İstanbul sea", {"sea"}),  # lower() змінює довжину тексту
    ("Привіт sea мир SEA", {"sea"}),  # кирилиця навколо ASCII-терма
    ("", {"sea"}),
    ("sea", set()),
]


@pytest.mark.parametrize(("text", "terms"), CASES)
def test_fast_path_equals_regex(text: str, terms: set[str]):
    assert _find_matches(text, terms) == _find_with_regex(text, terms)


def test_literal_scan_itself_equals_regex_on_ascii_text():
    text = "The Whale and the whales; a SHIP, a ship-yard, ships. Sea-sick at sea."
    terms = {"whale", "ship", "sea"}
    lowered = text.lower()
    assert _find_literals(text, lowered, terms) == _find_with_regex(text, terms)


@given(
    st.text(alphabet="abSAB _-'9éK ", max_size=60),
    st.sets(st.sampled_from(["a", "b", "ab", "ba", "aab", "s", "sa", "b-a", "é"]), max_size=3),
)
def test_property_fast_path_equals_regex(text: str, terms: set[str]):
    assert _find_matches(text, terms) == _find_with_regex(text, terms)


def test_same_matches_on_real_books():
    root = Path(__file__).resolve().parent.parent / "data" / "gutenberg"
    books = sorted(root.glob("*.txt"))[:6] if root.is_dir() else []
    if not books:
        pytest.skip("корпусу нема на диску")
    for path in books:
        text = strip_boilerplate(path.read_text(encoding="utf-8-sig"))
        for terms in ({"whale", "ship", "sea"}, {"the"}, {"love", "death"}):
            assert _find_matches(text, terms) == _find_with_regex(text, terms), path.name
