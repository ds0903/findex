import logging
from pathlib import Path

import pytest

from findex.documents import Document
from findex.index import STORAGES, build_index
from findex.query import QueryError
from findex.ranking import BM25, TfIdf
from findex.search import ENGINES
from findex.searcher import Searcher, search
from helpers import make_docs


def corpus(texts, **kwargs):
    return build_index(
        (
            Document(f"d{i}.txt", Path(f"d{i}.txt"), text, f"Doc {i}")
            for i, text in enumerate(texts)
        ),
        **kwargs,
    )


@pytest.fixture
def index():
    return build_index(make_docs())


def test_results_are_ranked_best_first(index):
    # doc 0 має cat і sat, doc 2 лише cat; "cat OR sat" - обидва підходять
    results = Searcher(index).search("cat OR sat")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.doc_id for r in results][0] == 0
    assert results[0].title == "Doc 0"


def test_only_matching_documents_are_returned(index):
    assert {r.doc_id for r in Searcher(index).search("cat dog")} == {2}
    assert Searcher(index).search("zzz") == []


def test_not_removes_documents_from_results(index):
    assert {r.doc_id for r in Searcher(index).search("cat NOT dog")} == {0}


def test_k_limits_results(index):
    searcher = Searcher(index)
    assert len(searcher.search("the OR cat OR dog", k=2)) == 2
    assert len(searcher.search("the OR cat OR dog", k=100)) == 3


def test_k_must_be_positive(index):
    with pytest.raises(ValueError):
        Searcher(index).search("cat", k=0)


def test_query_error_propagates(index):
    with pytest.raises(QueryError):
        Searcher(index).search("(cat")


def test_default_scorer_is_bm25(index):
    searcher = Searcher(index)
    assert searcher.search("cat") == searcher.search("cat", BM25())


def test_scorers_are_interchangeable(index):
    searcher = Searcher(index)
    bm25 = searcher.search("cat OR dog", BM25())
    tfidf = searcher.search("cat OR dog", TfIdf())
    assert {r.doc_id for r in bm25} == {r.doc_id for r in tfidf}


def test_query_with_only_not_returns_documents_with_zero_score(index):
    results = Searcher(index).search("NOT cat")
    assert [(r.doc_id, r.score) for r in results] == [(1, 0.0)]


def test_rare_word_ranks_document_first():
    index = corpus(["the the the cat", "the dog", "the harpoon", "the bird", "the fish"])
    results = Searcher(index).search("the OR harpoon")
    assert results[0].doc_id == 2


@pytest.mark.parametrize("storage", STORAGES)
def test_same_ranking_for_every_storage(storage):
    texts = ["cat dog dog", "cat", "dog bird cat cat", "bird"]
    reference = Searcher(corpus(texts)).search("cat OR dog")
    other = Searcher(corpus(texts, storage=storage)).search("cat OR dog")
    assert [(r.doc_id, round(r.score, 9)) for r in other] == [
        (r.doc_id, round(r.score, 9)) for r in reference
    ]


def test_engines_agree(index):
    merge = Searcher(index, engine="merge").search("(the OR a) NOT dog")
    other = Searcher(index, engine="set").search("(the OR a) NOT dog")
    assert merge == other
    assert set(ENGINES) == {"merge", "set"}


def test_phrase_search_with_positions():
    index = corpus(["event loop here", "loop event there", "event then loop"], positions=True)
    assert [r.doc_id for r in Searcher(index).search('"event loop"')] == [0]


def test_module_level_search_function(index):
    results = search(index, "cat", k=1)
    assert len(results) == 1


# --- кеш запитів ---


def test_repeated_query_is_a_cache_hit(index, caplog):
    searcher = Searcher(index)
    with caplog.at_level(logging.DEBUG, logger="findex.searcher"):
        searcher.search("cat dog")
        assert "cache hit" not in caplog.text
        assert searcher.cache_hits == 0
        searcher.search("cat dog")
    assert searcher.cache_hits == 1
    assert "cache hit: 'cat dog'" in caplog.text


def test_different_query_is_not_a_hit(index):
    searcher = Searcher(index)
    searcher.search("cat")
    searcher.search("dog")
    assert searcher.cache_hits == 0


def test_cache_is_per_searcher(index):
    first, second = Searcher(index), Searcher(index)
    first.search("cat")
    second.search("cat")
    assert (first.cache_hits, second.cache_hits) == (0, 0)


def test_cache_does_not_change_results(index):
    searcher = Searcher(index)
    assert searcher.search("cat OR dog") == searcher.search("cat OR dog")


def test_cache_size_is_limited(index):
    searcher = Searcher(index, cache_size=1)
    searcher.search("cat")
    searcher.search("dog")  # вичавлює "cat"
    searcher.search("cat")
    assert searcher.cache_hits == 0


def test_cached_value_is_immutable(index):
    matched = Searcher(index).matched("cat")
    assert isinstance(matched.doc_ids, tuple)
    assert matched.doc_ids == (0, 2)


def test_timed_logs_search_and_build(index, caplog):
    with caplog.at_level(logging.DEBUG):
        Searcher(index).search("cat")
        build_index(make_docs())
    assert "Searcher.search:" in caplog.text
    assert "build_index:" in caplog.text
    assert " мс" in caplog.text
