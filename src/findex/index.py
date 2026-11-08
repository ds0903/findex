"""Інвертований індекс: терм -> відсортований за doc_id список постінгів.

Індекс поводиться як звичайний об'єкт Python:

    len(index)          кількість термів
    "whale" in index    чи є такий терм
    index["whale"]      постінги терма (KeyError, якщо терма нема)
    for term in index   усі терми
"""

import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Protocol

from findex.documents import Document
from findex.timing import timed
from findex.tokenizer import tokenize

# як зберігати постінги (три варіанти для порівняння пам'яті):
#   slots - list[Posting], dataclass(frozen=True, slots=True)  (основний)
#   plain - list[PlainPosting], dataclass без slots
#   array - для кожного терма пара array('I'): doc_ids і tfs, жодних об'єктів
type Storage = Literal["slots", "plain", "array"]
STORAGES: tuple[Storage, ...] = ("slots", "plain", "array")


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int  # скільки разів терм зустрівся в документі
    positions: tuple[int, ...] = ()  # номери токенів; порожньо, якщо індекс без позицій


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


@dataclass(frozen=True)
class PlainPosting:
    """Те саме, що Posting, але без slots. Потрібен тільки для порівняння пам'яті."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


class PostingLike(Protocol):
    """Усе, що має doc_id, tf і positions: і Posting, і PlainPosting (без спільного предка)."""

    @property
    def doc_id(self) -> int: ...

    @property
    def tf(self) -> int: ...

    @property
    def positions(self) -> tuple[int, ...]: ...


# що лежить у словнику під одним термом
type Entry = list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]


@dataclass(repr=False)
class Index:
    postings: dict[str, Entry]
    doc_lengths: dict[int, int]  # doc_id -> кількість токенів
    doc_meta: dict[int, DocMeta]
    storage: Storage = "slots"
    positions: bool = False
    root: str = ""  # тека з корпусом, звідси беремо тексти для сніпетів

    # --- протоколи Python ---

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> Sequence[PostingLike]:
        entry = self.postings[term]  # немає терма - KeyError, як у dict
        if isinstance(entry, tuple):
            ids, tfs = entry
            return [Posting(d, tf) for d, tf in zip(ids, tfs, strict=True)]
        return entry

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    # --- документи ---

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        # cached_property: BM25 питає це число для кожного постінга, рахувати щоразу було б задорого
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def all_doc_ids(self) -> range:
        # doc_id видаються підряд з нуля
        return range(self.num_docs)

    # --- терми ---

    def df(self, term: str) -> int:
        """У скількох документах зустрічається терм."""
        entry = self.postings.get(term)
        if entry is None:
            return 0
        return len(entry[0]) if isinstance(entry, tuple) else len(entry)

    def doc_ids(self, term: str) -> Sequence[int]:
        """Відсортовані doc_id документів з цим термом (порожньо, якщо терма нема)."""
        entry = self.postings.get(term)
        if entry is None:
            return ()
        if isinstance(entry, tuple):
            return entry[0]
        return [p.doc_id for p in entry]

    def iter_postings(self, term: str) -> Iterator[PostingLike]:
        if term in self.postings:
            yield from self[term]

    def num_postings(self) -> int:
        return sum(self.df(term) for term in self.postings)

    def close(self) -> None:
        """Віддає пам'ять назад: після цього індекс порожній."""
        self.postings.clear()
        self.doc_lengths.clear()
        self.doc_meta.clear()
        self.__dict__.pop("avg_doc_length", None)


class _Sink(Protocol):
    def add(self, term: str, doc_id: int, tf: int, positions: tuple[int, ...]) -> None: ...

    def result(self) -> dict[str, Entry]: ...


class _SlotsSink:
    def __init__(self) -> None:
        self._data: defaultdict[str, list[Posting]] = defaultdict(list)

    def add(self, term: str, doc_id: int, tf: int, positions: tuple[int, ...]) -> None:
        self._data[term].append(Posting(doc_id, tf, positions))

    def result(self) -> dict[str, Entry]:
        # dict замість defaultdict, щоб пошук неіснуючого терма нічого не створював
        return {term: postings for term, postings in self._data.items()}


class _PlainSink:
    def __init__(self) -> None:
        self._data: defaultdict[str, list[PlainPosting]] = defaultdict(list)

    def add(self, term: str, doc_id: int, tf: int, positions: tuple[int, ...]) -> None:
        self._data[term].append(PlainPosting(doc_id, tf, positions))

    def result(self) -> dict[str, Entry]:
        return {term: postings for term, postings in self._data.items()}


def _new_arrays() -> tuple[array[int], array[int]]:
    return array("I"), array("I")


class _ArraySink:
    def __init__(self) -> None:
        self._data: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(_new_arrays)

    def add(self, term: str, doc_id: int, tf: int, positions: tuple[int, ...]) -> None:
        ids, tfs = self._data[term]
        ids.append(doc_id)
        tfs.append(tf)

    def result(self) -> dict[str, Entry]:
        return {term: pair for term, pair in self._data.items()}


def _make_sink(storage: Storage) -> _Sink:
    if storage == "slots":
        return _SlotsSink()
    if storage == "plain":
        return _PlainSink()
    return _ArraySink()


@timed
def build_index(
    documents: Iterable[Document],
    storage: Storage = "slots",
    positions: bool = False,
    root: str = "",
) -> Index:
    """Будує індекс за один прохід по документах.

    documents може бути генератором: тексти не накопичуються, в пам'яті росте
    лише сам індекс. Документи нумеруються по порядку, тому списки постінгів
    виходять відсортованими за doc_id самі собою.
    """
    if storage not in STORAGES:
        raise ValueError(f"невідоме сховище {storage!r}, є: {', '.join(STORAGES)}")
    if positions and storage == "array":
        raise ValueError("позиції поки що працюють лише зі сховищами slots і plain")

    sink = _make_sink(storage)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(documents):
        counts: Counter[str] = Counter()
        where: defaultdict[str, list[int]] = defaultdict(list)
        if positions:
            for pos, token in enumerate(tokenize(doc.text)):
                counts[token] += 1
                where[token].append(pos)
        else:
            counts.update(tokenize(doc.text))

        doc_lengths[doc_id] = counts.total()
        doc_meta[doc_id] = DocMeta(path=doc.doc_id, title=doc.title)

        # counts живе лише до кінця ітерації: з документа в індекс переходять тільки пари
        for term, tf in counts.items():
            sink.add(term, doc_id, tf, tuple(where[term]) if positions else ())

    return Index(sink.result(), doc_lengths, doc_meta, storage, positions, root)


if __name__ == "__main__":
    # Логіка CLI лежить в окремому модулі, а не тут: якби вона була в цьому файлі, то при
    # запуску через -m усі класи опинилися б у модулі __main__, і pickle потім
    # не зміг би знайти їх при завантаженні індексу з іншої програми.
    from findex.commands import index_main

    sys.exit(index_main())
