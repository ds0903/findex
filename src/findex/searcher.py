"""Ранжований пошук: запит -> документи -> рейтинг -> топ-k, з кешем запитів."""

import functools
import logging
from typing import Literal, NamedTuple

from findex.index import Index
from findex.query import Node, parse
from findex.ranking import BM25, Scorer, SearchResult, TfIdf, score_documents, top_k
from findex.search import Engine, evaluate_ids
from findex.timing import timed
from findex.vectorized import NumpyIndex, top_k_numpy

log = logging.getLogger(__name__)

Backend = Literal["python", "numpy"]  # не `type ...`: typer не розуміє аліаси з PEP 695
BACKEND_NAMES: tuple[Backend, ...] = ("python", "numpy")


class Matched(NamedTuple):
    node: Node
    doc_ids: tuple[int, ...]  # tuple, а не list: результат кешується й не має змінюватись


class Searcher:
    """Пошук по одному індексу.

    Шлях "текст запиту -> дерево + id документів" кешується через lru_cache. Кеш створюється
    для кожного Searcher окремо, а не на рівні модуля: ключем був би сам індекс, а Index
    (змінний об'єкт з порівнянням за значенням) нехешований, і не повинен бути хешованим.
    """

    def __init__(
        self,
        index: Index,
        engine: Engine = "merge",
        cache_size: int = 256,
        backend: Backend = "python",
    ) -> None:
        if backend not in BACKEND_NAMES:
            raise ValueError(f"невідомий backend {backend!r}, є: {', '.join(BACKEND_NAMES)}")
        self.index = index
        self.engine: Engine = engine
        self.backend: Backend = backend
        # масиви NumPy будуються один раз при створенні: це кілька секунд на весь корпус
        self._numpy = NumpyIndex.from_index(index) if backend == "numpy" else None
        self._match = functools.lru_cache(maxsize=cache_size)(self._match_uncached)

    def _match_uncached(self, query: str) -> Matched:
        node = parse(query)
        return Matched(node, tuple(evaluate_ids(self.index, node, self.engine)))

    @property
    def cache_hits(self) -> int:
        return self._match.cache_info().hits

    def matched(self, query: str) -> Matched:
        """Дерево запиту й усі doc_id, що йому відповідають (з кешу, якщо запит повторний)."""
        return self._match(query)

    @timed
    def search(self, query: str, scorer: Scorer | None = None, k: int = 10) -> list[SearchResult]:
        """k кращих документів за рейтингом (від кращого). Кидає QueryError, якщо запит поганий."""
        if k < 1:
            raise ValueError("k має бути не менше 1")
        scorer = scorer if scorer is not None else BM25()

        hits_before = self._match.cache_info().hits
        matched = self._match(query)
        if self._match.cache_info().hits > hits_before:
            log.debug("cache hit: %r", query)

        terms = matched.node.terms()
        # власний scorer користувача (Protocol) NumPy не знає: для нього лишається Python-шлях
        if self._numpy is not None and isinstance(scorer, BM25 | TfIdf):
            return top_k_numpy(self._numpy, terms, matched.doc_ids, scorer, k)
        scores = score_documents(self.index, terms, matched.doc_ids, scorer)
        return top_k(self.index, scores, k)


def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    engine: Engine = "merge",
) -> list[SearchResult]:
    """Разовий пошук без збереження кешу між викликами."""
    return Searcher(index, engine).search(query, scorer, k)
