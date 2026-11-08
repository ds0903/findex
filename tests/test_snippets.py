import logging

import pytest

from findex.index import DocMeta, Index
from findex.snippets import Snippet, document_text, make_snippet, snippet_for

LOREM = (
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no "
    "money in my purse, and nothing particular to interest me on shore, I thought I would "
    "sail about a little and see the watery part of the world. The white whale swam on."
)


def test_snippet_contains_match_and_is_limited():
    snippet = make_snippet(LOREM, {"whale"}, width=20)
    assert snippet is not None
    assert "whale" in snippet.text
    assert len(snippet.text) <= 2 * 20 + len("whale") + 6  # вікно + два "…" з пробілами


def test_highlight_marks_terms():
    snippet = make_snippet("the white whale swam", {"whale"}, width=80)
    assert snippet is not None
    assert snippet.highlight() == "the white **whale** swam"
    assert snippet.highlight("<b>", "</b>") == "the white <b>whale</b> swam"


def test_all_terms_are_highlighted_case_insensitive():
    snippet = make_snippet("Whale and WHALE and whale", {"whale"})
    assert snippet is not None
    assert snippet.highlight("[", "]") == "[Whale] and [WHALE] and [whale]"


def test_no_match_gives_none():
    assert make_snippet(LOREM, {"zzz"}) is None
    assert make_snippet(LOREM, set()) is None
    assert make_snippet("", {"a"}) is None


def test_does_not_match_inside_other_words():
    assert make_snippet("whales and whaler", {"whale"}) is None


def test_window_is_not_cut_in_the_middle_of_a_word():
    text = "x" * 30 + " alphabet whale " + "y" * 30
    snippet = make_snippet(text, {"whale"}, width=5)
    assert snippet is not None
    assert "alphabet" not in snippet.text or snippet.text.count("alphabet") == 1
    assert not any(part.startswith("lphabet") for part in snippet.text.split())


def test_ellipsis_only_where_text_was_cut():
    short = make_snippet("a whale here", {"whale"}, width=80)
    assert short is not None and "…" not in short.text

    middle = make_snippet(LOREM, {"whale"}, width=20)
    assert middle is not None
    assert middle.text.startswith("… ")


def test_best_window_has_most_distinct_terms():
    far = "whale " + "filler " * 100
    close = "the harpoon hit the whale hard"
    snippet = make_snippet(far + close, {"whale", "harpoon"}, width=40)
    assert snippet is not None
    assert "harpoon" in snippet.text and "whale" in snippet.text
    assert snippet.highlight().count("**") == 4  # обидва терми виділені


def test_newlines_become_spaces():
    snippet = make_snippet("first line\nthe whale\nlast line", {"whale"})
    assert snippet is not None
    assert "\n" not in snippet.text


def test_unicode_text_and_terms():
    snippet = make_snippet("Привіт, КОБЗАР Шевченка!", {"кобзар"})
    assert snippet is not None
    assert snippet.highlight() == "Привіт, **КОБЗАР** Шевченка!"


def test_combining_marks_are_normalized():
    snippet = make_snippet("a café here", {"café"})
    assert snippet is not None
    assert "**café**" in snippet.highlight()


def test_spans_point_into_text():
    snippet = make_snippet(LOREM, {"whale", "sail"}, width=60)
    assert snippet is not None
    assert snippet.spans
    for a, b in snippet.spans:
        assert snippet.text[a:b].casefold() in {"whale", "sail"}


def test_snippet_dataclass():
    assert Snippet("ab", ((0, 1),)).highlight("(", ")") == "(a)b"
    assert Snippet("ab", ()).highlight() == "ab"


# --- читання тексту з диска ---


@pytest.fixture
def index(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Title: A\n\n*** START OF THE PROJECT GUTENBERG EBOOK A ***\nthe white whale\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK A ***\nlicense whale junk\n",
        encoding="utf-8",
    )
    return Index(
        {},
        {0: 3, 1: 1},
        {0: DocMeta("a.txt", "A"), 1: DocMeta("gone.txt", "G")},
        root=str(tmp_path),
    )


def test_document_text_strips_boilerplate(index):
    text = document_text(index, 0)
    assert text is not None
    assert "white whale" in text
    assert "license" not in text


def test_snippet_for_reads_the_file(index):
    snippet = snippet_for(index, 0, {"whale"})
    assert snippet is not None
    assert snippet.highlight() == "the white **whale**"


def test_missing_file_gives_none_and_warning(index, caplog):
    with caplog.at_level(logging.WARNING):
        assert document_text(index, 1) is None
        assert snippet_for(index, 1, {"whale"}) is None
    assert "gone.txt" in caplog.text
