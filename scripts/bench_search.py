"""Бенчмарк merge проти set для лаби 2.

Запуск: python -u scripts/bench_search.py data/gutenberg
Беремо два найчастіших і два найрідших терми корпусу, плюс синтетичні списки
більшого розміру (в корпусі лише 59 документів, тож списки там короткі).
"""

import random
import sys
import timeit
from pathlib import Path

from findex.documents import iter_documents
from findex.index import build_index
from findex.measure import setup_console
from findex.search import merge_and, merge_diff, merge_or, set_and, set_diff, set_or

OPS = {
    "AND": (merge_and, set_and),
    "OR": (merge_or, set_or),
    "NOT": (merge_diff, set_diff),
}


def best_time(func, a, b) -> float:
    """Найкращий із 5 замірів, секунди на один виклик."""
    timer = timeit.Timer(lambda: func(a, b))
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=5, number=number)) / number


def row(label, op, a, b) -> str:
    merge_func, set_func = OPS[op]
    assert merge_func(a, b) == set_func(a, b)  # обидва рахують однаково
    t_merge = best_time(merge_func, a, b)
    t_set = best_time(set_func, a, b)
    winner = "merge" if t_merge < t_set else "set"
    ratio = max(t_merge, t_set) / min(t_merge, t_set)
    return (
        f"| {label} | {op} | {len(a)} | {len(b)} | {len(merge_func(a, b))} "
        f"| {t_merge * 1e6:.1f} | {t_set * 1e6:.1f} | {winner} (x{ratio:.1f}) |"
    )


def header(first: str) -> None:
    print(f"| {first} | операція | len(a) | len(b) | результат | merge, мкс | set, мкс | швидше |")
    print("|---|---|---:|---:|---:|---:|---:|---|")


def sorted_sample(rng: random.Random, universe: int, size: int) -> list[int]:
    return sorted(rng.sample(range(universe), size))


def main() -> None:
    setup_console()
    root = Path(sys.argv[1])

    print("будую індекс...", file=sys.stderr, flush=True)
    index = build_index(iter_documents(root))
    by_df = sorted(index.postings, key=lambda t: (-index.df(t), t))
    common = by_df[:2]
    rarest = sorted(t for t in index.postings if index.df(t) == 1)[:2]
    ids = index.doc_ids
    print(f"документів {len(index.doc_meta)}, термів {len(by_df)}\n")
    print(f"найчастіші: {[(t, index.df(t)) for t in common]}")
    print(f"найрідші (df=1, перші за абеткою): {[(t, index.df(t)) for t in rarest]}\n")

    print("## реальний корпус (списки doc_id з індексу)\n")
    header("терми")
    c1, c2 = common
    r1, r2 = rarest
    cases = [
        (f"{c1} / {c2}", "AND", ids(c1), ids(c2)),
        (f"{c1} / {c2}", "OR", ids(c1), ids(c2)),
        (f"{c1} / {c2}", "NOT", ids(c1), ids(c2)),
        (f"{r1} / {r2}", "AND", ids(r1), ids(r2)),
        (f"{r1} / {r2}", "OR", ids(r1), ids(r2)),
        (f"{c1} / {r1}", "AND", ids(c1), ids(r1)),
        (f"{c1} / {r1}", "OR", ids(c1), ids(r1)),
    ]
    for label, op, a, b in cases:
        print(row(label, op, a, b), flush=True)

    print("\n## синтетичні списки (випадкові відсортовані doc_id)\n")
    header("розмір")
    rng = random.Random(7)
    for n_a, n_b in [(1_000, 1_000), (100_000, 100_000), (100, 1_000_000)]:
        universe = max(n_a, n_b) * 5
        a = sorted_sample(rng, universe, n_a)
        b = sorted_sample(rng, universe, n_b)
        for op in ("AND", "OR"):
            print(row(f"{n_a} x {n_b}", op, a, b), flush=True)


if __name__ == "__main__":
    main()
