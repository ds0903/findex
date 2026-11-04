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
