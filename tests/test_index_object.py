import pytest

from findex.index import STORAGES, Index, Posting, build_index
from findex.store import open_index, save
from helpers import make_docs


@pytest.fixture(params=STORAGES)
def index(request):
    return build_index(make_docs(), storage=request.param)


def test_len_is_number_of_terms(index):
    assert len(index) == 8  # the cat sat on mat dog a and


def test_contains(index):
    assert "cat" in index
    assert "zzz" not in index


def test_getitem_returns_postings(index):
    postings = index["the"]
    assert [(p.doc_id, p.tf) for p in postings] == [(0, 2), (1, 1)]
    assert postings[0].positions == ()


def test_getitem_missing_term_is_key_error(index):
    with pytest.raises(KeyError):
        index["zzz"]


def test_iteration_gives_terms(index):
    assert set(index) == {"the", "cat", "sat", "on", "mat", "dog", "a", "and"}
    assert len(list(index)) == len(index)


def test_repr():
    assert repr(build_index(make_docs())) == "Index(terms=8, docs=3)"


def test_repr_separates_thousands():
    index = Index({}, {}, {})
    index.postings = {f"t{i}": [] for i in range(48_211)}
    assert repr(index) == "Index(terms=48_211, docs=0)"


def test_empty_index_is_falsy_because_len_is_zero():
    assert not build_index(iter(()))
    assert build_index(make_docs())


def test_num_docs_and_doc_length(index):
    assert index.num_docs == 3
    assert index.doc_length(0) == 6
    assert index.doc_length(1) == 3
    with pytest.raises(KeyError):
        index.doc_length(99)


def test_avg_doc_length_is_cached_property(index):
    assert "avg_doc_length" not in vars(index)  # поки не питали - не рахували
    assert index.avg_doc_length == pytest.approx((6 + 3 + 5) / 3)
    assert "avg_doc_length" in vars(index)

    index.doc_lengths[0] = 1000
    assert index.avg_doc_length == pytest.approx((6 + 3 + 5) / 3)  # береться з кешу


def test_avg_doc_length_of_empty_index():
    assert build_index(iter(())).avg_doc_length == 0.0


def test_df(index):
    assert index.df("the") == 2
    assert index.df("mat") == 1
    assert index.df("zzz") == 0


def test_iter_postings_of_unknown_term_is_empty(index):
    assert list(index.iter_postings("zzz")) == []


def test_array_storage_getitem_builds_posting_objects():
    index = build_index(make_docs(), storage="array")
    assert all(isinstance(p, Posting) for p in index["cat"])
    assert [(p.doc_id, p.tf) for p in index["cat"]] == [(0, 1), (2, 1)]


def test_close_releases_everything(index):
    _ = index.avg_doc_length
    index.close()
    assert len(index) == 0
    assert index.num_docs == 0
    assert "avg_doc_length" not in vars(index)


# --- open_index: контекстний менеджер ---


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "index.bin"
    save(build_index(make_docs()), path, "binary")
    return path


def test_open_index_yields_index_and_closes_it(saved):
    with open_index(saved) as index:
        assert len(index) == 8
        assert index.df("cat") == 2
    assert len(index) == 0  # після виходу пам'ять віддана


def test_open_index_closes_on_exception(saved):
    held = None
    with pytest.raises(RuntimeError, match="boom"):
        with open_index(saved) as index:
            held = index
            raise RuntimeError("boom")
    assert held is not None
    assert len(held) == 0


def test_open_index_closes_on_return(saved):
    def work():
        with open_index(saved) as index:
            return index

    assert len(work()) == 0


def test_open_index_missing_file_never_enters_with(tmp_path):
    entered = False
    with pytest.raises(FileNotFoundError):
        with open_index(tmp_path / "nope.bin"):
            entered = True
    assert not entered
