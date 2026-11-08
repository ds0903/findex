import math
from pathlib import Path

import pytest

from findex.documents import Document
from findex.index import Posting, build_index
from findex.ranking import BM25, Scorer, SearchResult, TfIdf, make_scorer, score_documents, top_k
from helpers import make_docs


def corpus(texts):
    return build_index(
        Document(f"d{i}.txt", Path(f"d{i}.txt"), text, f"Doc {i}") for i, text in enumerate(texts)
    )


# --- обидва scorer-и підходять під Protocol без наслідування ---


def test_both_scorers_satisfy_the_protocol():
    scorers: list[Scorer] = [TfIdf(), BM25()]
    index = build_index(make_docs())
    for scorer in scorers:
        assert scorer.score("cat", Posting(0, 1), index) >= 0
    assert not issubclass(BM25, TfIdf) and not issubclass(TfIdf, BM25)


def test_make_scorer():
    assert make_scorer("bm25") == BM25()
    assert make_scorer("tfidf") == TfIdf()
    with pytest.raises(ValueError):
        make_scorer("magic")  # type: ignore[arg-type]  # перевіряємо реакцію на поганий рядок


def test_bm25_defaults_and_repr():
    assert BM25() == BM25(k1=1.5, b=0.75)
    assert "k1=1.5" in repr(BM25())


# --- формули ---


def test_tfidf_formula():
    index = corpus(["x y", "x z", "y z", "w w w"])  # N = 4
    idf = math.log(4 / 1)  # "w" є в одному документі
    assert TfIdf().score("w", Posting(3, 3), index) == pytest.approx((1 + math.log(3)) * idf)


def test_tfidf_gives_zero_for_term_in_every_document():
    index = corpus(["x a", "x b", "x c"])
    assert TfIdf().score("x", Posting(0, 1), index) == 0.0


def test_bm25_formula():
    index = corpus(["x y", "x z", "y z", "w w w"])  # N = 4, avgdl = (2+2+2+3)/4
    k1, b = 1.5, 0.75
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    norm = 1 - b + b * 3 / index.avg_doc_length
    expected = idf * 3 * (k1 + 1) / (3 + k1 * norm)
    assert BM25().score("w", Posting(3, 3), index) == pytest.approx(expected)


def test_unknown_term_scores_zero():
    index = corpus(["x y"])
    assert BM25().score("zzz", Posting(0, 1), index) == 0.0
    assert TfIdf().score("zzz", Posting(0, 1), index) == 0.0


def test_bm25_idf_is_never_negative():
    index = corpus(["x a", "x b", "x c"])  # x є в усіх документах
    assert BM25().score("x", Posting(0, 1), index) > 0


# --- три перевірки з лаби: рідкий терм, 20-те повторення, коротший документ ---


def test_check1_rare_term_beats_common_term():
    index = corpus(["the cat", "the dog", "the bird", "the fish", "the cat harpoon", "the cow"])
    for scorer in (TfIdf(), BM25()):
        common = scorer.score("the", Posting(0, 1), index)
        rare = scorer.score("harpoon", Posting(4, 1), index)
        assert rare > common


def test_check2_twentieth_repetition_adds_almost_nothing_in_bm25():
    index = corpus(["x " * 50, "y " * 50, "z z", "w"])
    scorer = BM25()

    def s(tf):
        return scorer.score("x", Posting(0, tf), index)

    first_gain = s(2) - s(1)
    twentieth_gain = s(20) - s(19)
    assert twentieth_gain < first_gain / 20  # менше 5% від приросту на другому повторенні
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    assert s(1000) < idf * (scorer.k1 + 1)  # насичення: стеля idf * (k1 + 1)


def test_check2_tfidf_saturates_too_but_more_slowly():
    index = corpus(["x " * 50, "y " * 50, "z z", "w"])

    def gains(scorer: Scorer) -> tuple[float, float]:
        def s(tf: int) -> float:
            return scorer.score("x", Posting(0, tf), index)

        return s(2) - s(1), s(20) - s(19)

    tf_first, tf_twentieth = gains(TfIdf())
    bm_first, bm_twentieth = gains(BM25())
    assert tf_twentieth < tf_first / 10  # логарифм теж гасить ріст...
    assert tf_twentieth / tf_first > bm_twentieth / bm_first  # ...але BM25 гасить сильніше


def test_check3_short_document_beats_long_one_in_bm25():
    index = corpus(["needle", "needle " + "filler " * 200, "other words here"])
    short = BM25().score("needle", Posting(0, 1), index)
    long_ = BM25().score("needle", Posting(1, 1), index)
    assert short > long_
    # TF-IDF довжини не бачить: той самий tf - той самий рейтинг
    assert TfIdf().score("needle", Posting(0, 1), index) == TfIdf().score(
        "needle", Posting(1, 1), index
    )


def test_bm25_with_b_zero_ignores_length():
    index = corpus(["needle", "needle " + "filler " * 200, "other"])
    scorer = BM25(b=0)
    assert scorer.score("needle", Posting(0, 1), index) == scorer.score(
        "needle", Posting(1, 1), index
    )


# --- SearchResult і top-k ---


def test_search_result_positional_and_order():
    low, high = SearchResult(1, 0.5, "a"), SearchResult(2, 2.0, "b")
    assert low < high
    assert sorted([high, low]) == [low, high]
    assert max(low, high).doc_id == 2


def test_search_result_tie_prefers_smaller_doc_id():
    assert SearchResult(5, 1.0) < SearchResult(2, 1.0)  # doc 2 - "кращий"


def test_search_result_repr_hides_sort_key():
    assert "sort_key" not in repr(SearchResult(1, 1.0, "t"))


def test_top_k_returns_best_first_and_limits():
    index = build_index(make_docs())
    scores = {0: 1.0, 1: 3.0, 2: 2.0}
    assert [r.doc_id for r in top_k(index, scores, 2)] == [1, 2]
    assert [r.doc_id for r in top_k(index, scores, 10)] == [1, 2, 0]
    assert top_k(index, scores, 0) == []
    assert top_k(index, {}, 5) == []


def test_score_documents_sums_terms_over_matched_documents_only():
    index = build_index(make_docs())
    scores = score_documents(index, ["cat", "sat"], [0, 1], BM25())
    assert set(scores) == {0, 1}  # doc 2 не в множині збігів, хоч там є "cat"
    assert scores[0] > scores[1]  # doc 0 має обидва терми, doc 1 лише sat
