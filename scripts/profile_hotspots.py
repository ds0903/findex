"""Профілювання для лаби 8: де справді йде час.

    python scripts/profile_hotspots.py index data/gutenberg                 # cProfile побудови
    python scripts/profile_hotspots.py search data/index.bin data/gutenberg # cProfile пошуку
    py-spy record -o docs/flame_search.svg -- python scripts/profile_hotspots.py search ... --raw

Режим index будує індекс серійно (як `findex index --workers 1`). Режим search робить пошук із
трьох термів зі сніпетами для 10 результатів, так само, як `findex search`.
--raw вимикає cProfile (для py-spy, бо cProfile сам уповільнює код).
"""

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from findex.documents import iter_documents
from findex.index import build_index
from findex.ranking import BM25
from findex.searcher import Searcher
from findex.snippets import snippet_for
from findex.store import load

QUERY = "whale OR ship OR sea"
SEARCH_REPEAT = 5


def run_index(corpus: Path) -> None:
    build_index(iter_documents(corpus), root=str(corpus))


def run_search(searcher: Searcher, k: int) -> None:
    results = searcher.search(QUERY, BM25(), k)
    terms = searcher.matched(QUERY).node.terms()
    for r in results:
        snippet_for(searcher.index, r.doc_id, terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["index", "search"])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--raw", action="store_true", help="без cProfile (для py-spy)")
    parser.add_argument("--backend", choices=["python", "numpy"], default="python")
    parser.add_argument("--top", type=int, default=14)
    parser.add_argument("--no-snippets", action="store_true")
    args = parser.parse_args()

    if args.mode == "index":

        def job() -> None:
            run_index(args.paths[0])

    else:
        index = load(args.paths[0])
        index.root = str(args.paths[1])
        searcher = Searcher(index, cache_size=0, backend=args.backend)

        def job() -> None:
            for _ in range(SEARCH_REPEAT):
                if args.no_snippets:
                    searcher.search(QUERY, BM25(), 10)
                else:
                    run_search(searcher, 10)

    if args.raw:
        started = time.perf_counter()
        job()
        print(f"{time.perf_counter() - started:.2f} с")
        return

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.runcall(job)
    print(f"загалом з cProfile: {time.perf_counter() - started:.2f} с")
    for sort in ("tottime", "cumulative"):
        out = io.StringIO()
        stats = pstats.Stats(profiler, stream=out).strip_dirs().sort_stats(sort)
        stats.print_stats(args.top)
        print(f"\n--- за {sort} ---")
        print(out.getvalue().strip())


if __name__ == "__main__":
    main()
