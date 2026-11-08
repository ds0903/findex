from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.documents import iter_documents
from findex.index import STORAGES, Index, build_index
from findex.query import parse
from findex.ranking import BM25, TfIdf, score_documents, top_k
from findex.search import evaluate_ids
from findex.searcher import Searcher
from findex.vectorized import (
    NumpyIndex,
    gap_decode,
    gap_encode,
    score_array,
    top_k_numpy,
)
from helpers import make_docs

QUERIES = [
    "sea",
    "whale",
    "the",
    "ship AND sea",
    "whale OR ship",
    "sea AND NOT whale",
    "cat OR dog OR mat",
    "dog",
    "nothing-here",
]


def python_result(index: Index, query: str, scorer, k: int):
    node = parse(query)
    ids = evaluate_ids(index, node)
    return top_k(index, score_documents(index, node.terms(), ids, scorer), k)


def numpy_result(npi: NumpyIndex, index: Index, query: str, scorer, k: int):
    node = parse(query)
    return top_k_numpy(npi, node.terms(), evaluate_ids(index, node), scorer, k)


def same(a, b) -> bool:
    """Той самий порядок, ті самі документи й рівно ті самі числа (не "майже")."""
    return [(r.doc_id, r.score, r.title) for r in a] == [(r.doc_id, r.score, r.title) for r in b]


@pytest.fixture
def real_index(corpus: Path) -> Index:
    return build_index(iter_documents(corpus), root=str(corpus))


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("scorer", [BM25(), BM25(k1=1.2, b=0.3), TfIdf()])
@pytest.mark.parametrize("k", [1, 2, 10])
def test_same_ranking_as_lab_3(storage: str, scorer, k: int):
    index = build_index(
        make_docs(
            [
                "the cat sat on the mat",
                "the dog sat",
                "a cat and a dog",
                "cat cat cat dog",
                "the the the the the cat",
                "nothing at all here",
            ]
        ),
        storage=storage,  # type: ignore[arg-type]
    )
    npi = NumpyIndex.from_index(index)
    for query in ["cat", "the", "cat AND dog", "cat OR dog", "cat AND NOT dog", "sat OR mat OR a"]:
        assert same(
            python_result(index, query, scorer, k), numpy_result(npi, index, query, scorer, k)
        )


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_same_ranking_on_fixture_corpus(real_index: Index, scorer):
    npi = NumpyIndex.from_index(real_index)
    for query in QUERIES:
        for k in (1, 5, 10):
            assert same(
                python_result(real_index, query, scorer, k),
                numpy_result(npi, real_index, query, scorer, k),
            ), (query, k)


def test_same_ranking_on_the_real_corpus():
    """Весь корпус з лаби 1 (59 книг), якщо він є на диску: порядок і числа збігаються."""
    root = Path(__file__).resolve().parent.parent / "data" / "gutenberg"
    if not root.is_dir():
        pytest.skip("корпусу нема на диску")
    index = build_index(iter_documents(root), root=str(root))
    npi = NumpyIndex.from_index(index)
    for query in ["whale", "white AND whale", "love OR death", "the", "ship AND sea AND storm"]:
        for scorer in (BM25(), TfIdf()):
            assert same(
                python_result(index, query, scorer, 20), numpy_result(npi, index, query, scorer, 20)
            ), query


def test_ties_are_broken_by_doc_id_even_when_cut_by_k():
    texts = ["whale ship"] * 8 + ["whale"]  # вісім однакових документів і один інший
    index = build_index(make_docs(texts))
    npi = NumpyIndex.from_index(index)
    for k in (1, 3, 5, 8):
        got = numpy_result(npi, index, "whale AND ship", BM25(), k)
        assert [r.doc_id for r in got] == list(range(k))  # рівні за рейтингом: раніший id кращий


def test_empty_results_and_unknown_terms():
    index = build_index(make_docs())
    npi = NumpyIndex.from_index(index)
    assert top_k_numpy(npi, ["cat"], [], BM25(), 5) == []
    assert numpy_result(npi, index, "zzz", BM25(), 5) == []
    assert npi.df("zzz") == 0 and npi.df("cat") == 2


def test_not_query_gives_zero_scores_like_lab_3():
    index = build_index(make_docs())
    npi = NumpyIndex.from_index(index)
    assert same(
        python_result(index, "NOT cat", BM25(), 5), numpy_result(npi, index, "NOT cat", BM25(), 5)
    )


def test_custom_scorer_is_rejected_by_numpy_but_searcher_falls_back():
    class Constant:
        def score(self, term, posting, index):
            return 1.0

    index = build_index(make_docs())
    npi = NumpyIndex.from_index(index)
    with pytest.raises(TypeError, match="лише BM25 і TfIdf"):
        score_array(npi, ["cat"], Constant())
    searcher = Searcher(index, backend="numpy")
    assert [r.score for r in searcher.search("cat", Constant(), 5)] == [1.0, 1.0]


def test_arrays_are_int32(real_index: Index):
    npi = NumpyIndex.from_index(real_index)
    assert npi.doc_ids.dtype == npi.tfs.dtype == npi.doc_lengths.dtype == np.int32
    assert len(npi.doc_ids) == real_index.num_postings()
    ids, tfs = npi.postings("sea")
    assert list(ids) == list(real_index.doc_ids("sea"))
    assert [p.tf for p in real_index["sea"]] == list(tfs)


def test_searcher_backends_agree(real_index: Index):
    python = Searcher(real_index)
    fast = Searcher(real_index, backend="numpy")
    for query in QUERIES:
        for scorer in (BM25(), TfIdf()):
            assert same(python.search(query, scorer, 5), fast.search(query, scorer, 5))
    with pytest.raises(ValueError, match="невідомий backend"):
        Searcher(real_index, backend="gpu")  # type: ignore[arg-type]


@given(
    st.lists(
        st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=1, max_size=12),
        min_size=1,
        max_size=15,
    ),
    st.lists(st.sampled_from(["a", "b", "c", "d", "e", "zz"]), min_size=1, max_size=3),
    st.integers(1, 10),
    st.sampled_from(["bm25", "tfidf"]),
)
def test_property_numpy_equals_python(texts, terms, k, which):
    index = build_index(make_docs([" ".join(words) for words in texts]))
    npi = NumpyIndex.from_index(index)
    scorer = BM25() if which == "bm25" else TfIdf()
    ids = evaluate_ids(index, parse(" OR ".join(terms)))
    expected = top_k(index, score_documents(index, terms, ids, scorer), k)
    assert same(expected, top_k_numpy(npi, terms, ids, scorer, k))


# --- збереження й memmap ---


def test_save_and_load_with_memmap(tmp_path: Path, real_index: Index):
    npi = NumpyIndex.from_index(real_index)
    npi.save(tmp_path / "npy")
    mapped = NumpyIndex.load(tmp_path / "npy", mmap=True)
    loaded = NumpyIndex.load(tmp_path / "npy", mmap=False)
    assert isinstance(mapped.doc_ids, np.memmap) and not isinstance(loaded.doc_ids, np.memmap)
    for query in QUERIES:
        node = parse(query)
        ids = evaluate_ids(real_index, node)
        want = top_k_numpy(npi, node.terms(), ids, BM25(), 5)
        assert same(want, top_k_numpy(mapped, node.terms(), ids, BM25(), 5))
        assert same(want, top_k_numpy(loaded, node.terms(), ids, BM25(), 5))


# --- стиснення різницями ---


def test_gap_encoding_roundtrip(real_index: Index):
    npi = NumpyIndex.from_index(real_index)
    gaps, firsts = gap_encode(npi.doc_ids, npi.offsets)
    assert gaps.dtype == np.uint16 and gaps.nbytes == npi.doc_ids.nbytes // 2
    for i, (term, (start, end)) in enumerate(npi.offsets.items()):
        assert list(gap_decode(gaps, int(firsts[i]), start, end)) == list(npi.postings(term)[0])


@given(st.lists(st.sets(st.integers(0, 5000), min_size=1, max_size=30), min_size=1, max_size=8))
def test_gap_encoding_property(lists):
    flat: list[int] = []
    offsets: dict[str, tuple[int, int]] = {}
    for i, ids in enumerate(lists):
        offsets[f"t{i}"] = (len(flat), len(flat) + len(ids))
        flat.extend(sorted(ids))
    doc_ids = np.array(flat, dtype=np.int32)
    gaps, firsts = gap_encode(doc_ids, offsets)
    for i, (start, end) in enumerate(offsets.values()):
        assert list(gap_decode(gaps, int(firsts[i]), start, end)) == flat[start:end]


def test_gap_encoding_rejects_too_large_gaps():
    ids = np.array([0, 100_000], dtype=np.int32)
    with pytest.raises(ValueError, match="uint16"):
        gap_encode(ids, {"t": (0, 2)})


def test_gap_decode_of_empty_list():
    assert len(gap_decode(np.zeros(0, dtype=np.uint16), 0, 0, 0)) == 0
