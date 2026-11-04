"""Інвертований індекс: терм -> відсортований за doc_id список постінгів.

Запуск: python -m findex.index data/ --out index.bin
"""

import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from findex.documents import Document
from findex.tokenizer import tokenize

# як зберігати постінги (три варіанти для порівняння пам'яті):
#   slots - list[Posting], dataclass(frozen=True, slots=True)  (основний)
#   plain - list[PlainPosting], dataclass без slots
#   array - для кожного терма пара array('I'): doc_ids і tfs, жодних об'єктів
STORAGES = ("slots", "plain", "array")


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


@dataclass
class Index:
    # slots/plain: терм -> list[Posting]; array: терм -> (array doc_ids, array tfs)
    postings: dict
    doc_lengths: dict[int, int]  # doc_id -> кількість токенів
    doc_meta: dict[int, DocMeta]
    storage: str = "slots"
    positions: bool = False

    def doc_ids(self, term: str) -> Sequence[int]:
        """Відсортовані doc_id документів з цим термом (порожньо, якщо терма нема)."""
        entry = self.postings.get(term)
        if entry is None:
            return ()
        if self.storage == "array":
            return entry[0]
        return [p.doc_id for p in entry]

    def iter_postings(self, term: str) -> Iterator[Posting | PlainPosting]:
        entry = self.postings.get(term)
        if entry is None:
            return
        if self.storage == "array":
            for doc_id, tf in zip(entry[0], entry[1], strict=True):
                yield Posting(doc_id, tf)
        else:
            yield from entry

    def df(self, term: str) -> int:
        """У скількох документах зустрічається терм."""
        entry = self.postings.get(term)
        if entry is None:
            return 0
        return len(entry[0]) if self.storage == "array" else len(entry)

    def all_doc_ids(self) -> range:
        # doc_id видаються підряд з нуля
        return range(len(self.doc_meta))

    def num_postings(self) -> int:
        return sum(self.df(term) for term in self.postings)


def _new_arrays() -> tuple[array, array]:
    return array("I"), array("I")


def build_index(
    documents: Iterable[Document], storage: str = "slots", positions: bool = False
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

    postings = defaultdict(_new_arrays if storage == "array" else list)
    posting_cls = PlainPosting if storage == "plain" else Posting
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
            if storage == "array":
                ids, tfs = postings[term]
                ids.append(doc_id)
                tfs.append(tf)
            else:
                where_term = tuple(where[term]) if positions else ()
                postings[term].append(posting_cls(doc_id, tf, where_term))

    # dict замість defaultdict, щоб пошук неіснуючого терма нічого не створював
    return Index(dict(postings), doc_lengths, doc_meta, storage, positions)


if __name__ == "__main__":
    # Логіка CLI лежить у commands.py, а не тут: якби вона була в цьому файлі, то при
    # запуску через -m усі класи опинилися б у модулі __main__, і pickle потім
    # не зміг би знайти їх при завантаженні індексу з іншої програми.
    from findex.commands import index_main

    sys.exit(index_main())
