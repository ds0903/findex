"""Булів пошук по інвертованому індексу: які документи підходять (без рейтингу).

Запит розбирає findex.query (AND, OR, NOT, дужки, "фрази"). Дерево можна виконати
двома способами (прапорець engine):

    merge - двовказівниковий прохід по відсортованих спискам, O(len(a) + len(b))
    set   - через множини Python: швидкий C-код, але треба будувати множини й сортувати

Ранжований пошук зі сніпетами: findex.searcher. Запуск: python -m findex.search index.bin "query"
"""

import sys
from collections.abc import Callable, Sequence
from typing import Literal

from findex.index import Index
from findex.query import And, Node, Not, Or, Phrase, Term, parse

type Engine = Literal["merge", "set"]
ENGINE_NAMES: tuple[Engine, ...] = ("merge", "set")

type Combine = Callable[[Sequence[int], Sequence[int]], list[int]]

# --- двовказівниковий merge: списки вже відсортовані, тому за один прохід ---


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])  # один із списків закінчився, решта іншого йде як є
    out.extend(b[j:])
    return out


def merge_diff(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Елементи a, яких нема в b."""
    out: list[int] = []
    i = j = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


# --- те саме через set: швидко пише C, але треба будувати множини й потім сортувати ---


def set_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_diff(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


ENGINES: dict[Engine, tuple[Combine, Combine, Combine]] = {
    "merge": (merge_and, merge_or, merge_diff),
    "set": (set_and, set_or, set_diff),
}


def evaluate_ids(index: Index, node: Node, engine: Engine = "merge") -> list[int]:
    """Відсортований список doc_id, що задовольняють дерево запиту."""
    if engine not in ENGINES:
        raise ValueError(f"невідомий engine {engine!r}, є: {', '.join(ENGINE_NAMES)}")
    and_, or_, diff = ENGINES[engine]

    def go(n: Node) -> Sequence[int]:
        match n:
            case Term(text):
                return index.doc_ids(text)
            case Phrase():
                return sorted(n.evaluate(index))
            case And(left, right):
                return and_(go(left), go(right))
            case Or(left, right):
                return or_(go(left), go(right))
            case Not(operand):
                return diff(index.all_doc_ids(), go(operand))
            case _:
                raise TypeError(f"невідомий вузол запиту: {n!r}")

    return list(go(node))


def search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    """Розбирає запит і повертає відсортований список doc_id (без рейтингу)."""
    return evaluate_ids(index, parse(query), engine)


if __name__ == "__main__":
    # див. пояснення в index.py: CLI тримаємо в окремому модулі
    from findex.commands import search_main

    sys.exit(search_main())
