"""CLI для `python -m findex.index` і `python -m findex.search`.

Окремий модуль, щоб класи індексу завжди жили в findex.index (а не в __main__),
інакше pickle не зможе завантажити збережений індекс. Деталі - в index.py.
"""

import argparse
import logging
import sys
import time
from itertools import islice
from pathlib import Path

from findex.documents import iter_documents
from findex.index import STORAGES, build_index
from findex.measure import mb, measure, setup_console
from findex.search import ENGINES, search
from findex.store import FORMATS, load, save


def index_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.index", description="будує інвертований індекс і зберігає його"
    )
    parser.add_argument("root", type=Path, help="тека з .txt файлами")
    parser.add_argument("--out", type=Path, required=True, help="куди зберегти індекс")
    parser.add_argument("--format", choices=FORMATS, default="binary")
    parser.add_argument("--storage", choices=STORAGES, default="slots")
    parser.add_argument("--positions", action="store_true", help="зберігати позиції токенів")
    parser.add_argument("--limit", type=int, default=None, help="взяти лише перші N документів")
    parser.add_argument(
        "--no-memory", action="store_true", help="не вмикати tracemalloc (час точніший)"
    )
    args = parser.parse_args(argv)

    setup_console()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    def job():
        start = time.perf_counter()
        docs = islice(iter_documents(args.root), args.limit)
        index = build_index(docs, storage=args.storage, positions=args.positions)
        built = time.perf_counter() - start
        start = time.perf_counter()
        save(index, args.out, args.format)
        return index, built, time.perf_counter() - start

    try:
        m = measure(job, trace_memory=not args.no_memory)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    index, built, saved = m.result
    print(f"документів:   {len(index.doc_meta)}")
    print(f"термів:       {len(index.postings)}")
    print(f"постінгів:    {index.num_postings()}")
    print(f"сховище:      {index.storage}, позиції: {'так' if index.positions else 'ні'}")
    print(f"побудова:     {built:.2f} с")
    size = mb(args.out.stat().st_size)
    print(f"збереження:   {saved:.2f} с -> {args.out} ({size:.1f} МБ, {args.format})")
    if m.peak_bytes is None:
        print("пік пам'яті:  не мірялась")
    else:
        print(f"пік пам'яті:  {mb(m.peak_bytes):.1f} МБ (tracemalloc, тому час більший)")
    return 0


def search_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.search", description="шукає по збереженому індексу"
    )
    parser.add_argument("index", type=Path, help="файл, створений findex.index")
    parser.add_argument("query", help='запит, наприклад "whale OR ship NOT sea"')
    parser.add_argument("--engine", choices=tuple(ENGINES), default="merge")
    parser.add_argument("--limit", type=int, default=10, help="скільки результатів показати")
    parser.add_argument(
        "--no-memory", action="store_true", help="не вмикати tracemalloc (час точніший)"
    )
    args = parser.parse_args(argv)

    setup_console()

    def job():
        start = time.perf_counter()
        index = load(args.index)
        loaded = time.perf_counter() - start
        start = time.perf_counter()
        hits = search(index, args.query, args.engine)
        return index, hits, loaded, time.perf_counter() - start

    try:
        m = measure(job, trace_memory=not args.no_memory)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    index, hits, loaded, searched = m.result
    print(f"запит: {args.query!r}, engine: {args.engine}")
    print(f"знайдено документів: {len(hits)}")
    for doc_id in hits[: args.limit]:
        meta = index.doc_meta[doc_id]
        print(f"  {doc_id:>4}  {meta.title}  ({meta.path})")
    if len(hits) > args.limit:
        print(f"  ... і ще {len(hits) - args.limit}")
    print(f"завантаження: {loaded:.3f} с")
    print(f"пошук:        {searched * 1000:.3f} мс")
    if m.peak_bytes is None:
        print("пік пам'яті:  не мірялась")
    else:
        print(f"пік пам'яті:  {mb(m.peak_bytes):.1f} МБ (tracemalloc)")
    return 0
