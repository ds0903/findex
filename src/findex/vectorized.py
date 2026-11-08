"""Рейтинг виразами над масивами NumPy (лаба 8).

Постінги терма це два масиви int32: doc_id і кількість входжень. Довжини документів це один
масив int32. BM25 і TF-IDF рахуються для всього списку постінгів одним виразом, а топ-k береться
через np.argpartition: O(n) замість повного сортування.

Порядок результатів збігається з лабою 3 (є тест), для цього:
- формули записані в тому самому порядку дій, що й у ranking.py (IEEE-арифметика детермінована);
- логарифми беруться з math.log (np.log може відрізнятися в останньому біті): idf рахується
  скаляром, а ln(tf) береться з таблиці, заповненої math.log;
- внески термів додаються в порядку термів запиту, як у score_documents;
- при однаковому рейтингу кращий документ з меншим doc_id, як у SearchResult.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.ranking import BM25, Scorer, SearchResult, TfIdf

I32 = npt.NDArray[np.int32]
F64 = npt.NDArray[np.float64]


@dataclass
class NumpyIndex:
    """Плоскі масиви: постінги всіх термів підряд, offsets[term] = (початок, кінець)."""

    offsets: dict[str, tuple[int, int]]
    doc_ids: I32  # усі постінги підряд, у межах терма за зростанням
    tfs: I32
    doc_lengths: I32  # int32[N]
    titles: list[str]
    _log_tf: F64 | None = field(default=None, repr=False)

    # --- побудова ---

    @classmethod
    def from_index(cls, index: Index) -> "NumpyIndex":
        offsets: dict[str, tuple[int, int]] = {}
        id_parts: list[I32] = []
        tf_parts: list[I32] = []
        start = 0
        for term in index:
            entry = index.postings[term]
            if isinstance(entry, tuple):  # сховище "array": два array('i'), буфер без копіювання
                ids = np.asarray(entry[0], dtype=np.int32)
                tfs = np.asarray(entry[1], dtype=np.int32)
            else:
                ids = np.fromiter((p.doc_id for p in entry), dtype=np.int32, count=len(entry))
                tfs = np.fromiter((p.tf for p in entry), dtype=np.int32, count=len(entry))
            offsets[term] = (start, start + len(ids))
            start += len(ids)
            id_parts.append(ids)
            tf_parts.append(tfs)
        n = index.num_docs
        lengths = np.fromiter((index.doc_lengths[i] for i in range(n)), dtype=np.int32, count=n)
        titles = [index.doc_meta[i].title for i in range(n)]
        empty = np.zeros(0, dtype=np.int32)
        return cls(
            offsets,
            np.concatenate(id_parts) if id_parts else empty,
            np.concatenate(tf_parts) if tf_parts else empty,
            lengths,
            titles,
        )

    # --- доступ ---

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        # цілочисельна сума й одне ділення, як в Index.avg_doc_length: число збігається до біта
        total = int(self.doc_lengths.sum(dtype=np.int64))
        return total / self.num_docs if self.num_docs else 0.0

    def postings(self, term: str) -> tuple[I32, I32]:
        start, end = self.offsets[term]
        return self.doc_ids[start:end], self.tfs[start:end]

    def df(self, term: str) -> int:
        span = self.offsets.get(term)
        return 0 if span is None else span[1] - span[0]

    @property
    def log_tf(self) -> F64:
        """ln(tf) для tf = 0..max: таблиця, заповнена math.log (див. докстрінг модуля)."""
        if self._log_tf is None:
            top = int(self.tfs.max()) if len(self.tfs) else 1
            table = np.zeros(top + 1, dtype=np.float64)
            for value in range(1, top + 1):
                table[value] = math.log(value)
            self._log_tf = table
        return self._log_tf

    # --- збереження: окремі .npy файли, які можна відкрити через memmap ---

    def save(self, directory: Path | str) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "doc_ids.npy", self.doc_ids)
        np.save(out / "tfs.npy", self.tfs)
        np.save(out / "doc_lengths.npy", self.doc_lengths)
        meta = {"offsets": self.offsets, "titles": self.titles}
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str, mmap: bool = True) -> "NumpyIndex":
        """mmap=True: масиви не читаються з диска, а відображаються в пам'ять (np.memmap)."""
        source = Path(directory)
        mode = "r" if mmap else None
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        return cls(
            {term: (a, b) for term, (a, b) in meta["offsets"].items()},
            np.load(source / "doc_ids.npy", mmap_mode=mode),
            np.load(source / "tfs.npy", mmap_mode=mode),
            np.load(source / "doc_lengths.npy"),
            meta["titles"],
        )


# --- рейтинг ---


def term_contribution(npi: NumpyIndex, term: str, scorer: Scorer) -> tuple[I32, F64] | None:
    """(doc_id, внесок терма) для всіх документів із термом; None, якщо терма нема."""
    df = npi.df(term)
    n = npi.num_docs
    if df == 0 or n == 0:
        return None
    ids, tfs = npi.postings(term)
    if isinstance(scorer, BM25):
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        length_norm = 1 - scorer.b + scorer.b * npi.doc_lengths[ids] / npi.avg_doc_length
        return ids, idf * tfs * (scorer.k1 + 1) / (tfs + scorer.k1 * length_norm)
    if isinstance(scorer, TfIdf):
        idf = math.log(n / df)
        return ids, (1 + npi.log_tf[tfs]) * idf
    raise TypeError(f"для NumPy є лише BM25 і TfIdf, не {type(scorer).__name__}")


def score_array(npi: NumpyIndex, terms: Sequence[str], scorer: Scorer) -> F64:
    """Рейтинг усіх документів (float64[N]); внески додаються в порядку термів запиту."""
    scores = np.zeros(npi.num_docs, dtype=np.float64)
    for term in terms:
        part = term_contribution(npi, term, scorer)
        if part is not None:
            ids, values = part
            scores[ids] += values  # у межах терма doc_id унікальні, тож це не втрачає додавань
    return scores


def top_k_numpy(
    npi: NumpyIndex,
    terms: Sequence[str],
    doc_ids: Sequence[int],
    scorer: Scorer,
    k: int,
) -> list[SearchResult]:
    """k кращих серед doc_ids, від кращого: рейтинг спадає, при рівності doc_id зростає."""
    if not doc_ids:
        return []
    candidates = np.asarray(doc_ids, dtype=np.int64)
    scores = score_array(npi, terms, scorer)[candidates]
    if len(candidates) > k:
        top = np.argpartition(-scores, k - 1)[:k]  # k кращих, але в довільному порядку
        threshold = scores[top].min()
        # усі, хто не гірший за k-й: інакше серед рівних на межі argpartition вибрав би випадкових
        pool = np.flatnonzero(scores >= threshold)
    else:
        pool = np.arange(len(candidates))
    order = np.lexsort((candidates[pool], -scores[pool]))[:k]  # перший ключ: -рейтинг, другий: id
    return [
        SearchResult(int(candidates[i]), float(scores[i]), npi.titles[int(candidates[i])])
        for i in pool[order]
    ]


# --- стиснення: різниці між сусідніми doc_id ---


def gap_encode(
    doc_ids: I32, offsets: dict[str, tuple[int, int]]
) -> tuple[npt.NDArray[np.uint16], I32]:
    """Кожен список зберігається як перший id і різниці до попереднього (uint16, якщо влазить).

    Повертає (різниці як uint16 для всіх постінгів, перші id кожного терма в порядку offsets).
    Перші id лишаються окремо (int32), а в масиві різниць на їхньому місці стоїть 0.
    Кидає ValueError, якщо різниця не влазить у uint16.
    """
    gaps = np.zeros(len(doc_ids), dtype=np.int64)
    firsts = np.zeros(len(offsets), dtype=np.int32)
    for i, (start, end) in enumerate(offsets.values()):
        if end > start:
            firsts[i] = doc_ids[start]
            gaps[start + 1 : end] = np.diff(doc_ids[start:end].astype(np.int64))
    if len(gaps) and gaps.max() > np.iinfo(np.uint16).max:
        raise ValueError("різниця між doc_id не влазить у uint16")
    return gaps.astype(np.uint16), firsts


def gap_decode(gaps: npt.NDArray[np.uint16], first: int, start: int, end: int) -> I32:
    """Відновлює doc_id одного терма: перший id + накопичена сума різниць (np.cumsum)."""
    if end <= start:
        return np.zeros(0, dtype=np.int32)
    part = gaps[start:end].astype(np.int64)
    part[0] = first
    return np.cumsum(part).astype(np.int32)
