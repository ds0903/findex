import pickle

import pytest

from findex.documents import Document
from findex.index import STORAGES, build_index
from findex.store import FORMATS, MAGIC, load, save
from helpers import make_docs

CASES = [
    (storage, positions)
    for storage in STORAGES
    for positions in (False, True)
    if not (storage == "array" and positions)
]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(("storage", "positions"), CASES)
def test_round_trip(tmp_path, fmt, storage, positions):
    original = build_index(make_docs(), storage=storage, positions=positions)
    path = tmp_path / "index.bin"
    save(original, path, fmt)

    loaded = load(path)
    assert loaded == original
    assert loaded.storage == storage
    assert loaded.positions == positions


@pytest.mark.parametrize("fmt", FORMATS)
def test_unicode_titles_and_terms_survive(tmp_path, fmt):
    docs = [
        Document(
            doc_id="a.txt", path=tmp_path / "a.txt", text="Привіт світе, café", title="Кобзар"
        ),
    ]
    path = tmp_path / "idx"
    save(build_index(docs), path, fmt)
    loaded = load(path)
    assert loaded.doc_meta[0].title == "Кобзар"
    assert set(loaded.postings) == {"привіт", "світе", "café"}


def test_format_is_detected_by_content_not_extension(tmp_path):
    index = build_index(make_docs())
    save(index, tmp_path / "a.pkl", "binary")  # розширення бреше
    save(index, tmp_path / "b.bin", "pickle")
    assert load(tmp_path / "a.pkl") == index
    assert load(tmp_path / "b.bin") == index


def test_binary_file_starts_with_magic(tmp_path):
    save(build_index(make_docs()), tmp_path / "i.bin", "binary")
    assert (tmp_path / "i.bin").read_bytes().startswith(MAGIC)


def test_truncated_binary_file(tmp_path):
    path = tmp_path / "i.bin"
    save(build_index(make_docs()), path, "binary")
    data = path.read_bytes()
    path.write_bytes(data[:-5])
    with pytest.raises(ValueError):
        load(path)


def test_broken_header(tmp_path):
    path = tmp_path / "i.bin"
    path.write_bytes(MAGIC + b"\x05\x00\x00\x00{not json")
    with pytest.raises(ValueError):
        load(path)


def test_garbage_file(tmp_path):
    path = tmp_path / "junk"
    path.write_bytes(b"this is not an index at all")
    with pytest.raises(ValueError):
        load(path)


def test_pickle_with_something_else_inside_is_rejected(tmp_path):
    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps({"not": "an index"}))
    with pytest.raises(ValueError):
        load(path)


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        save(build_index(make_docs()), tmp_path / "x", "yaml")


def test_missing_file(tmp_path):
    with pytest.raises(OSError):
        load(tmp_path / "nope.bin")
