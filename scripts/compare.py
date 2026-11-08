"""Порівняння lazy і eager версій на різній кількості документів.

Запуск: uv run python scripts/compare.py data/gutenberg [N ...]
Для кожної пари (версія, N) робимо два запуски: без tracemalloc (чесний час)
і з tracemalloc (піковий обсяг пам'яті), бо tracemalloc сильно сповільнює код.
Виводить таблицю в markdown.
"""

import sys
from itertools import islice
from pathlib import Path

from findex.documents import iter_documents
from findex.measure import setup_console
from findex.stats import collect_stats, collect_stats_eager, measure


def run(root: Path, limit: int | None, eager: bool):
    collect = collect_stats_eager if eager else collect_stats

    def job():
        return collect(islice(iter_documents(root), limit))

    timed = measure(job, trace_memory=False)
    traced = measure(job, trace_memory=True)
    return timed, traced


def main() -> None:
    setup_console()
    root = Path(sys.argv[1])
    limits = [int(x) for x in sys.argv[2:]] or [10, 25, 50, None]

    print("| версія | документів | токенів | словник | пік пам'яті, МБ | час, с |")
    print("|---|---:|---:|---:|---:|---:|")
    for limit in limits:
        for eager in (False, True):
            timed, traced = run(root, limit, eager)
            s = traced.stats
            name = "eager" if eager else "lazy"
            peak = traced.peak_bytes or 0  # с trace_memory=True значення завжди є
            print(
                f"| {name} | {s.documents} | {s.tokens} | {s.vocabulary} "
                f"| {peak / 1024 / 1024:.1f} | {timed.elapsed:.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
