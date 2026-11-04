import argparse
import logging
import sys
from itertools import islice
from pathlib import Path

from findex.documents import iter_documents
from findex.stats import Measurement, collect_stats, collect_stats_eager, measure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="findex", description="findex: лаба 1")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="статистика корпусу за один прохід")
    stats.add_argument("root", type=Path, help="тека з .txt файлами")
    stats.add_argument("--limit", type=int, default=None, help="взяти лише перші N документів")
    stats.add_argument("--top", type=int, default=50, help="скільки топ-термів показати")
    stats.add_argument("--eager", action="store_true", help="жадібна версія (через списки)")
    stats.add_argument(
        "--no-memory",
        action="store_true",
        help="не вмикати tracemalloc (тоді час точніший, але пам'ять не міряється)",
    )
    return parser


def print_report(m: Measurement, eager: bool, top_n: int) -> None:
    s = m.stats
    print(f"версія:       {'eager (списки)' if eager else 'lazy (генератори)'}")
    print(f"документів:   {s.documents}")
    print(f"токенів:      {s.tokens}")
    print(f"словник:      {s.vocabulary}")
    if m.peak_bytes is None:
        print(f"час:          {m.elapsed:.2f} с")
        print("пік пам'яті:  не мірялась")
    else:
        print(f"час:          {m.elapsed:.2f} с (з tracemalloc, тому повільніше)")
        print(f"пік пам'яті:  {m.peak_bytes / 1024 / 1024:.1f} МБ")
    print(f"\nтоп-{top_n} термів:")
    for rank, (term, count) in enumerate(s.top_terms, start=1):
        print(f"{rank:>4}. {term:<20} {count}")


def main(argv: list[str] | None = None) -> int:
    # у windows консоль часто cp1251, а в корпусі є будь-які символи
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args = build_parser().parse_args(argv)

    def run():
        docs = islice(iter_documents(args.root), args.limit)
        collect = collect_stats_eager if args.eager else collect_stats
        return collect(docs, top_n=args.top)

    try:
        result = measure(run, trace_memory=not args.no_memory)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print_report(result, args.eager, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
