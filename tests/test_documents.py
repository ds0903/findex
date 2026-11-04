import inspect
import logging

import pytest

from findex.documents import extract_title, iter_documents, strip_boilerplate


def test_is_generator(tmp_path):
    assert inspect.isgenerator(iter_documents(tmp_path))


def test_reads_lazily(tmp_path):
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")

    docs = iter_documents(tmp_path)
    first = next(docs)
    assert first.doc_id == "a.txt"
    assert first.text == "first"

    # поки не запросили, другий файл ще не прочитаний: видалимо його
    (tmp_path / "b.txt").unlink()
    assert list(docs) == []


def test_order_is_stable_and_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "skip.md").write_text("no", encoding="utf-8")

    ids = [d.doc_id for d in iter_documents(tmp_path)]
    assert ids == ["a.txt", "b.txt", "sub/c.txt"]


def test_bad_file_is_skipped_and_logged(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00\xd8 not utf-8 \xc3\x28")
    (tmp_path / "c.txt").write_text("ok too", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        ids = [d.doc_id for d in iter_documents(tmp_path)]

    assert ids == ["a.txt", "c.txt"]
    assert "bad.txt" in caplog.text


def test_missing_root_raises_on_first_next(tmp_path):
    gen = iter_documents(tmp_path / "nope")  # помилки ще нема - генератор не запускався
    with pytest.raises(FileNotFoundError):
        next(gen)


def test_strip_boilerplate():
    raw = (
        "license junk\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "real text\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "more junk\n"
    )
    assert strip_boilerplate(raw).strip() == "real text"
    assert strip_boilerplate("no markers here") == "no markers here"


def test_extract_title():
    raw = "The Project Gutenberg eBook of X\n\nTitle: Moby Dick; Or, The Whale  \nAuthor: H. M.\n"
    assert extract_title(raw) == "Moby Dick; Or, The Whale"
    assert extract_title("no header", default="fallback") == "fallback"


def test_document_has_title_from_header_even_after_strip(tmp_path):
    (tmp_path / "pg1.txt").write_text(
        "Title: Some Book\n\n*** START OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\ntext\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\n",
        encoding="utf-8",
    )
    (tmp_path / "plain.txt").write_text("just text", encoding="utf-8")

    docs = {d.doc_id: d for d in iter_documents(tmp_path)}
    assert docs["pg1.txt"].title == "Some Book"
    assert docs["pg1.txt"].text.strip() == "text"
    assert docs["plain.txt"].title == "plain"  # нема Title - беремо ім'я файлу
