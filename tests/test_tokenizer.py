import inspect

import pytest

from findex.tokenizer import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_is_generator():
    assert inspect.isgenerator(tokenize("hello world"))


def test_empty_and_whitespace():
    assert toks("") == []
    assert toks("  \n\t ") == []
    assert toks("... --- !!!") == []


def test_mixed_case_is_folded():
    assert toks("Hello WORLD hElLo") == ["hello", "world", "hello"]


def test_casefold_not_just_lower():
    assert toks("Straße") == ["strasse"]


def test_cyrillic():
    assert toks("Привіт, СВІТЕ! Їжак") == ["привіт", "світе", "їжак"]


def test_combining_marks_are_normalized():
    # "é" одним символом і як e + U+0301 - це має бути один і той самий токен
    assert toks("café") == toks("café") == ["café"]
    # й = и + U+0306
    assert toks("й") == ["й"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("don't", ["don't"]),
        ("don’t", ["don't"]),  # типографський апостроф
        ("м’ясо п'ять", ["м'ясо", "п'ять"]),
        ("'hello'", ["hello"]),  # лапки по краях не токен
        ("dogs' bones", ["dogs", "bones"]),
    ],
)
def test_apostrophes(text, expected):
    assert toks(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("well-known", ["well-known"]),
        ("well‐known", ["well-known"]),  # unicode-дефіс
        ("end- start", ["end", "start"]),
        ("-leading", ["leading"]),
        ("one -- two", ["one", "two"]),
        ("one — two", ["one", "two"]),
        ("2020-2021", ["2020-2021"]),
    ],
)
def test_hyphens_and_dashes(text, expected):
    assert toks(text) == expected


def test_digits():
    assert toks("covid19 and 1984") == ["covid19", "and", "1984"]
    assert toks("pi is 3.14") == ["pi", "is", "3", "14"]
    assert toks("1,000") == ["1", "000"]


def test_underscore_separates_words():
    # у Gutenberg так позначений курсив
    assert toks("this is _very_ good") == ["this", "is", "very", "good"]


def test_punctuation_only_between_words():
    assert toks("Hello,world;foo.bar") == ["hello", "world", "foo", "bar"]
