import pytest

from findex.documents import iter_documents
from findex.stats import collect_stats, collect_stats_eager, measure


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.txt").write_text("The cat sat. The cat ran!", encoding="utf-8")
    (tmp_path / "b.txt").write_text("A dog; the dog barked.", encoding="utf-8")
    (tmp_path / "c.txt").write_text("Кіт і пес, кіт", encoding="utf-8")
    return tmp_path


def test_counts(corpus):
    stats = collect_stats(iter_documents(corpus))
    assert stats.documents == 3
    assert stats.tokens == 6 + 5 + 4
    # the cat sat ran a dog barked кіт і пес
    assert stats.vocabulary == 10
    assert stats.top_terms[0] == ("the", 3)


def test_top_n_limit(corpus):
    stats = collect_stats(iter_documents(corpus), top_n=2)
    assert len(stats.top_terms) == 2


def test_empty_corpus(tmp_path):
    stats = collect_stats(iter_documents(tmp_path))
    assert (stats.documents, stats.tokens, stats.vocabulary) == (0, 0, 0)
    assert stats.top_terms == []


def test_eager_matches_lazy(corpus):
    lazy = collect_stats(iter_documents(corpus))
    eager = collect_stats_eager(iter_documents(corpus))
    assert lazy == eager


def test_measure_reports_time_and_peak(corpus):
    m = measure(lambda: collect_stats(iter_documents(corpus)))
    assert m.elapsed >= 0
    assert m.peak_bytes is not None and m.peak_bytes > 0

    m = measure(lambda: collect_stats(iter_documents(corpus)), trace_memory=False)
    assert m.peak_bytes is None
