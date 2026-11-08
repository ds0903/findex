import pytest

from findex.commands import index_main, search_main

BOOKS = {
    "a.txt": "Title: Whale Book\n\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
    "the white whale swims in the sea\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n",
    "b.txt": "Title: Ship Book\n\n*** START OF THE PROJECT GUTENBERG EBOOK Y ***\n"
    "the ship sails the sea\n*** END OF THE PROJECT GUTENBERG EBOOK Y ***\n",
}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    for name, text in BOOKS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def built(corpus, tmp_path, capsys):
    out = tmp_path / "index.bin"
    assert index_main([str(corpus), "--out", str(out), "--no-memory", "--positions"]) == 0
    capsys.readouterr()
    return out


@pytest.mark.parametrize("fmt", ["binary", "pickle"])
def test_build_then_search(corpus, tmp_path, capsys, fmt):
    out = tmp_path / "index.bin"
    assert index_main([str(corpus), "--out", str(out), "--format", fmt]) == 0
    built = capsys.readouterr().out
    assert "документів:   2" in built
    assert "пік пам'яті" in built
    assert out.exists()

    assert search_main([str(out), "sea NOT whale"]) == 0
    found = capsys.readouterr().out
    assert "знайдено документів: 1" in found
    assert "Ship Book" in found
    assert "Whale Book" not in found
    assert "пошук:" in found
    assert "пік пам'яті" in found


def test_results_have_scores_and_highlighted_snippets(built, capsys):
    assert search_main([str(built), "whale", "--no-memory"]) == 0
    out = capsys.readouterr().out
    assert "  1." in out
    assert "Whale Book  (a.txt)" in out
    assert "the white **whale** swims" in out


def test_no_snippets_flag(built, capsys):
    assert search_main([str(built), "whale", "--no-snippets", "--no-memory"]) == 0
    assert "**" not in capsys.readouterr().out


def test_phrase_and_parentheses(built, capsys):
    # "the sea" є в обох книгах, тож дужки з ship/whale вирішують лише з NOT
    assert search_main([str(built), '"the sea" AND (ship OR whale)', "--no-memory"]) == 0
    assert "знайдено документів: 2" in capsys.readouterr().out

    query = '"the sea" AND (ship OR whale) NOT whale'
    assert search_main([str(built), query, "--no-memory"]) == 0
    out = capsys.readouterr().out
    assert "знайдено документів: 1" in out
    assert "Ship Book" in out and "Whale Book" not in out


def test_scorer_and_engine_options(built, capsys):
    for scorer in ("bm25", "tfidf"):
        for engine in ("merge", "set"):
            argv = [str(built), "sea", "--scorer", scorer, "--engine", engine, "--no-memory"]
            assert search_main(argv) == 0
            assert f"scorer: {scorer}, engine: {engine}" in capsys.readouterr().out


def test_k_limits_shown_results(built, capsys):
    assert search_main([str(built), "sea", "-k", "1", "--no-memory", "--no-snippets"]) == 0
    out = capsys.readouterr().out
    assert "знайдено документів: 2" in out
    assert "  1." in out and "  2." not in out


def test_verbose_shows_timing_and_cache_hit_in_stderr(built, capsys):
    argv = [str(built), "sea", "--repeat", "2", "-vv", "--no-memory"]
    assert search_main(argv) == 0
    err = capsys.readouterr().err
    assert "Searcher.search:" in err
    assert "cache hit: 'sea'" in err
    assert "load:" in err


def test_without_verbose_stderr_is_quiet(built, capsys):
    assert search_main([str(built), "sea", "--repeat", "2", "--no-memory"]) == 0
    assert capsys.readouterr().err == ""


def test_bad_query_is_one_line_error(built, capsys):
    assert search_main([str(built), "(sea", "--no-memory"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("помилка:")
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_search_with_set_engine_and_no_memory(corpus, tmp_path, capsys):
    out = tmp_path / "index.bin"
    index_main([str(corpus), "--out", str(out), "--no-memory"])
    capsys.readouterr()
    assert search_main([str(out), "the sea", "--engine", "set", "--no-memory"]) == 0
    found = capsys.readouterr().out
    assert "знайдено документів: 2" in found
    assert "не мірялась" in found


def test_index_options(corpus, tmp_path, capsys):
    out = tmp_path / "index.bin"
    assert index_main([str(corpus), "--out", str(out), "--storage", "plain", "--positions"]) == 0
    assert "сховище:      plain, позиції: так" in capsys.readouterr().out
    assert index_main([str(corpus), "--out", str(out), "--limit", "1"]) == 0
    assert "документів:   1" in capsys.readouterr().out


def test_missing_corpus_dir(tmp_path, capsys):
    assert index_main([str(tmp_path / "nope"), "--out", str(tmp_path / "i.bin")]) == 1


def test_search_in_missing_or_broken_file(tmp_path, capsys):
    assert search_main([str(tmp_path / "nope.bin"), "x"]) == 1
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"junk")
    assert search_main([str(junk), "x"]) == 1
