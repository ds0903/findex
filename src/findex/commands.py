"""CLI для `python -m findex.index` і `python -m findex.search`.

Окремий модуль, щоб класи індексу завжди жили в findex.index (а не в __main__),
інакше pickle не зможе завантажити збережений індекс. Деталі - в index.py.
"""

import argparse
import sys
import time
from itertools import islice
from pathlib import Path

from findex.documents import iter_documents
from findex.index import STORAGES, build_index
from findex.logconfig import setup_logging
from findex.measure import mb, measure, setup_console
from findex.query import QueryError
from findex.ranking import SCORER_NAMES, make_scorer
from findex.search import ENGINE_NAMES
from findex.searcher import Searcher
from findex.snippets import snippet_for
from findex.store import FORMATS, open_index, save


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
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    setup_console()
    setup_logging(args.verbose)

    def job():
        start = time.perf_counter()
        docs = islice(iter_documents(args.root), args.limit)
        index = build_index(
            docs,
            storage=args.storage,
            positions=args.positions,
            root=str(args.root.resolve()),
        )
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
    print(f"документів:   {index.num_docs}")
    print(f"термів:       {len(index)}")
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
        prog="python -m findex.search", description="ранжований пошук по збереженому індексу"
    )
    parser.add_argument("index", type=Path, help="файл, створений findex.index")
    parser.add_argument("query", help='запит, наприклад: whale AND (harpoon OR ship) NOT "sea"')
    parser.add_argument("--scorer", choices=SCORER_NAMES, default="bm25")
    parser.add_argument("--engine", choices=ENGINE_NAMES, default="merge")
    parser.add_argument("-k", "--k", type=int, default=10, help="скільки результатів показати")
    parser.add_argument("--no-snippets", action="store_true", help="не читати тексти для сніпетів")
    parser.add_argument("--repeat", type=int, default=1, help="виконати запит N разів (кеш)")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG (час і кеш)"
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="не вмикати tracemalloc (час точніший)"
    )
    args = parser.parse_args(argv)

    setup_console()
    setup_logging(args.verbose)
    scorer = make_scorer(args.scorer)

    def job():
        start = time.perf_counter()
        with open_index(args.index) as index:
            loaded = time.perf_counter() - start
            searcher = Searcher(index, engine=args.engine)
            start = time.perf_counter()
            for _ in range(args.repeat):
                results = searcher.search(args.query, scorer, args.k)
            searched = time.perf_counter() - start
            lines: list[str] = []
            matched = searcher.matched(args.query)  # дерево вже в кеші
            terms = matched.node.terms()
            for rank, r in enumerate(results, start=1):
                path = index.doc_meta[r.doc_id].path
                lines.append(f"{rank:>3}. {r.score:7.3f}  {r.title}  ({path})")
                if not args.no_snippets:
                    snippet = snippet_for(index, r.doc_id, terms)
                    if snippet is not None:
                        lines.append(f"     {snippet.highlight()}")
            return lines, len(matched.doc_ids), loaded, searched / args.repeat

    try:
        m = measure(job, trace_memory=not args.no_memory)
    except (OSError, ValueError, QueryError) as exc:
        print(f"помилка: {exc}", file=sys.stderr)
        return 1

    lines, found, loaded, searched = m.result
    print(f"запит: {args.query!r}, scorer: {args.scorer}, engine: {args.engine}")
    print(f"знайдено документів: {found}")
    for line in lines:
        print(line)
    print(f"завантаження: {loaded:.3f} с")
    print(f"пошук:        {searched * 1000:.3f} мс")
    if m.peak_bytes is None:
        print("пік пам'яті:  не мірялась")
    else:
        print(f"пік пам'яті:  {mb(m.peak_bytes):.1f} МБ (tracemalloc)")
    return 0
