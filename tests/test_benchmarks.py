"""Бенчмарки пошуку (pytest-benchmark): python проти numpy на синтетичному індексі.

Звичайний запуск пропускає їх (pytest --benchmark-skip), у CI є окремий крок:
    pytest tests/test_benchmarks.py --benchmark-only
"""

from pathlib import Path

import numpy as np
import pytest

from findex.documents import Document
from findex.index import Index, build_index
from findex.ranking import BM25
from findex.searcher import Searcher

pytest.importorskip("pytest_benchmark")

DOCS = 4000
QUERIES = {"common": "w1", "rare": "w4000", "three_terms": "w1 OR w50 OR w4000"}


def documents() -> list[Document]:
    rng = np.random.default_rng(8)
    out: list[Document] = []
    for i in range(DOCS):
        ranks = np.minimum(rng.zipf(1.3, 60), 5000)
        out.append(Document(f"s{i}", Path(f"s{i}.txt"), " ".join(f"w{r}" for r in ranks), f"d{i}"))
    return out


@pytest.fixture(scope="module")
def index() -> Index:
    return build_index(documents(), storage="array")


@pytest.fixture(scope="module")
def searchers(index: Index) -> dict[str, Searcher]:
    return {
        "python": Searcher(index, cache_size=0, backend="python"),
        "numpy": Searcher(index, cache_size=0, backend="numpy"),
    }


@pytest.mark.parametrize("backend", ["python", "numpy"])
@pytest.mark.parametrize("name", list(QUERIES))
def test_search_speed(benchmark, searchers: dict[str, Searcher], backend: str, name: str):
    query = QUERIES[name]
    results = benchmark(searchers[backend].search, query, BM25(), 10)
    assert len(results) == 10


@pytest.mark.parametrize("name", list(QUERIES))
def test_backends_agree_on_benchmark_data(searchers: dict[str, Searcher], name: str):
    query = QUERIES[name]
    python = [(r.doc_id, r.score) for r in searchers["python"].search(query, BM25(), 10)]
    fast = [(r.doc_id, r.score) for r in searchers["numpy"].search(query, BM25(), 10)]
    assert python == fast
