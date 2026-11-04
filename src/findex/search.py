"""Булів пошук по інвертованому індексу.

Запит: `term term` - це AND, також є OR і NOT (великими літерами), обчислюється зліва направо:

    cat dog            cat AND dog
    cat OR dog         cat або dog
    cat NOT dog        cat, але без dog
    NOT cat            усі документи, де cat нема
    a OR b NOT c       (a OR b) NOT c

Запуск: python -m findex.search index.bin "query"
"""

import sys
from collections.abc import Sequence

from findex.index import Index
from findex.tokenizer import tokenize

OPERATORS = {"AND", "OR", "NOT"}


# --- двовказівниковий merge: списки вже відсортовані, тому за один прохід, O(len(a) + len(b)) ---


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out = []
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
    out = []
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
    out = []
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


ENGINES = {
    "merge": (merge_and, merge_or, merge_diff),
    "set": (set_and, set_or, set_diff),
}


def search(index: Index, query: str, engine: str = "merge") -> list[int]:
    """Повертає відсортований список doc_id, що задовольняють запит."""
    if engine not in ENGINES:
        raise ValueError(f"невідомий engine {engine!r}, є: {', '.join(ENGINES)}")
    and_, or_, diff = ENGINES[engine]
    everything = index.all_doc_ids()

    result: Sequence[int] | None = None
    op = "AND"
    negate = False
    for word in query.split():
        if word in OPERATORS:
            if word == "NOT":
                negate = True
            else:
                op = word
            continue
        # одне слово запиту може дати кілька термів ("foo.bar"), вони йдуть як AND
        for term in tokenize(word):
            ids = index.doc_ids(term)
            if result is None:
                result = diff(everything, ids) if negate else ids
            elif negate:
                result = diff(result, ids) if op == "AND" else or_(result, diff(everything, ids))
            else:
                result = and_(result, ids) if op == "AND" else or_(result, ids)
            op, negate = "AND", False

    return list(result) if result else []


if __name__ == "__main__":
    # див. пояснення в index.py: CLI тримаємо в commands.py
    from findex.commands import search_main

    sys.exit(search_main())
