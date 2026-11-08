"""Ранжування: TF-IDF і BM25 як взаємозамінні scorer-и, топ-k через heapq.

Scorer - це typing.Protocol: підходить будь-який об'єкт з методом score(...) потрібної форми,
наслідувати щось не треба (структурна типізація, "качина" типізація, але перевірена).
"""

import heapq
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal, Protocol

from findex.index import Index, PostingLike

type ScorerName = Literal["bm25", "tfidf"]
SCORER_NAMES: tuple[ScorerName, ...] = ("bm25", "tfidf")


class Scorer(Protocol):
    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        """Внесок одного терма в рейтинг документа posting.doc_id."""
        ...


@dataclass(frozen=True)
class TfIdf:
    """score = (1 + ln tf) * ln(N / df). Терм, який є в усіх документах, нічого не дає."""

    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        df = index.df(term)
        if df == 0 or index.num_docs == 0:
            return 0.0
        idf = math.log(index.num_docs / df)
        return (1 + math.log(posting.tf)) * idf


@dataclass(frozen=True)
class BM25:
    """Okapi BM25 з idf як у Lucene (завжди додатний).

    idf   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    k1 обмежує внесок повторень (насичення), b - наскільки сильно штрафувати довгі
    документи: при b = 0 довжина не враховується взагалі.
    """

    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        df = index.df(term)
        n = index.num_docs
        if df == 0 or n == 0:
            return 0.0
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        avg = index.avg_doc_length
        length_norm = 1 - self.b + self.b * index.doc_length(posting.doc_id) / avg
        tf = posting.tf
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)


def make_scorer(name: ScorerName) -> Scorer:
    if name == "bm25":
        return BM25()
    if name == "tfidf":
        return TfIdf()
    raise ValueError(f"невідомий scorer {name!r}, є: {', '.join(SCORER_NAMES)}")


@dataclass(order=True)
class SearchResult:
    """Результат пошуку. sorted(results) впорядковує від гіршого до кращого:
    за рейтингом, а при рівності - раніший doc_id вважається кращим."""

    sort_key: tuple[float, int] = field(init=False, repr=False)
    doc_id: int = field(compare=False)
    score: float = field(compare=False)
    title: str = field(compare=False, default="")

    def __post_init__(self) -> None:
        self.sort_key = (self.score, -self.doc_id)


def score_documents(
    index: Index, terms: Collection[str], doc_ids: Collection[int], scorer: Scorer
) -> dict[int, float]:
    """Сумує внески термів запиту по документах із doc_ids."""
    wanted = set(doc_ids)
    scores = dict.fromkeys(wanted, 0.0)
    for term in terms:
        for posting in index.iter_postings(term):
            if posting.doc_id in wanted:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    return scores


def top_k(index: Index, scores: dict[int, float], k: int) -> list[SearchResult]:
    """k кращих документів, від кращого до гіршого. nlargest тримає купу з k елементів:
    O(n log k), тоді як sorted(...)[:k] - O(n log n) і копія всього списку."""
    results = (
        SearchResult(doc_id, score, index.doc_meta[doc_id].title)
        for doc_id, score in scores.items()
    )
    return heapq.nlargest(k, results)
