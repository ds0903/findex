"""Командний рядок findex: `findex index`, `findex search`, `findex stats`.

Правила для всіх команд:
- результат (таблиця, JSON, підсумок) іде в stdout, усе діагностичне (логи, прогрес) у stderr;
- логування налаштовується тут і лише тут (`-v` це INFO, `-vv` це DEBUG), бібліотечний код
  тільки пише в свої логери;
- очікувані помилки (немає файлу, поганий запит, пошкоджений індекс) це один рядок у stderr
  і ненульовий код виходу, без traceback.
"""

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from findex.documents import find_paths, iter_documents
from findex.index import Index, Storage, build_index
from findex.logconfig import setup_logging
from findex.measure import mb, measure, setup_console
from findex.ranking import ScorerName, make_scorer
from findex.search import Engine
from findex.searcher import Searcher
from findex.snippets import snippet_for
from findex.stats import Stats, collect_stats, collect_stats_eager
from findex.store import Format, load, open_index, save

log = logging.getLogger(__name__)

app = typer.Typer(
    name="findex",
    help="Пошукова система: будує інвертований індекс і шукає в ньому з рейтингом BM25/TF-IDF.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,  # у locals лежав би весь індекс
)

Verbose = Annotated[
    int,
    typer.Option("--verbose", "-v", count=True, help="-v: INFO, -vv: DEBUG (у stderr)"),
]


def _show_version(value: bool) -> None:
    if not value:
        return
    try:
        typer.echo(f"findex {version('findex')}")
    except PackageNotFoundError:
        typer.echo("findex (не встановлено як пакет)")
    raise typer.Exit()


@app.callback()
def main(
    show_version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="показати версію"),
    ] = False,
) -> None:
    pass


@contextmanager
def expected_errors() -> Generator[None]:
    """Очікувані помилки -> один рядок у stderr і код виходу 1. Решта (баги) падає як є."""
    try:
        yield
    except (OSError, ValueError) as exc:  # QueryError і помилки load теж ValueError
        log.debug("деталі помилки", exc_info=True)
        typer.echo(f"помилка: {' '.join(str(exc).split())}", err=True)
        raise typer.Exit(1) from exc


def _progress() -> Progress:
    console = Console(stderr=True)  # прогрес у stderr, щоб не псувати stdout
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not console.is_terminal,  # у пайпі чи файлі смужка не потрібна
    )


# --- findex index ---


@app.command("index")
def index_command(
    root: Annotated[Path, typer.Argument(help="тека з .txt файлами")],
    out: Annotated[Path, typer.Option("--out", "-o", help="куди зберегти індекс")],
    fmt: Annotated[Format, typer.Option("--format", help="формат файлу")] = "binary",
    storage: Annotated[Storage, typer.Option("--storage", help="як тримати постінги")] = "slots",
    positions: Annotated[bool, typer.Option("--positions", help="зберігати позиції")] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="лише перші N")] = None,
    no_memory: Annotated[bool, typer.Option("--no-memory", help="без tracemalloc")] = False,
    verbose: Verbose = 0,
) -> None:
    """Будує індекс з усіх .txt у теці й зберігає його у файл."""
    setup_console()
    setup_logging(verbose)

    with expected_errors():
        paths = find_paths(root)
        if not paths:
            raise ValueError(f"у {root} немає .txt файлів")
        total = len(paths) if limit is None else min(limit, len(paths))
        log.info("файлів до індексації: %d", total)

        def job() -> tuple[Index, float, float]:
            with _progress() as progress:
                docs = progress.track(
                    islice(iter_documents(root), limit), total=total, description="індексація"
                )
                start = time.perf_counter()
                index = build_index(docs, storage, positions, root=str(root.resolve()))
                built = time.perf_counter() - start
            start = time.perf_counter()
            save(index, out, fmt)
            return index, built, time.perf_counter() - start

        measured = measure(job, trace_memory=not no_memory)

    index, built, saved = measured.result
    log.info("індекс збережено в %s", out)
    typer.echo(f"документів:   {index.num_docs}")
    typer.echo(f"термів:       {len(index)}")
    typer.echo(f"постінгів:    {index.num_postings()}")
    typer.echo(f"сховище:      {index.storage}, позиції: {'так' if index.positions else 'ні'}")
    typer.echo(f"побудова:     {built:.2f} с")
    typer.echo(f"збереження:   {saved:.2f} с -> {out} ({mb(out.stat().st_size):.1f} МБ, {fmt})")
    if measured.peak_bytes is None:
        typer.echo("пік пам'яті:  не мірялась")
    else:
        typer.echo(
            f"пік пам'яті:  {mb(measured.peak_bytes):.1f} МБ (tracemalloc, тому час більший)"
        )


# --- findex search ---


@app.command("search")
def search_command(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="файл від `findex index`")],
    query: Annotated[str, typer.Argument(help='запит: whale AND (harpoon OR ship) NOT "sea"')],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="скільки результатів")] = 10,
    scorer: Annotated[ScorerName, typer.Option("--scorer", help="формула рейтингу")] = "bm25",
    engine: Annotated[Engine, typer.Option("--engine", help="булів пошук")] = "merge",
    as_json: Annotated[bool, typer.Option("--json", help="JSON lines у stdout")] = False,
    no_snippets: Annotated[bool, typer.Option("--no-snippets", help="без уривків")] = False,
    repeat: Annotated[int, typer.Option("--repeat", min=1, help="N разів (кеш, -vv)")] = 1,
    verbose: Verbose = 0,
) -> None:
    """Ранжований пошук по збереженому індексу."""
    setup_console()
    setup_logging(verbose)

    rows: list[dict[str, object]] = []
    highlighted: list[Text | None] = []
    with expected_errors(), open_index(index_path) as index:
        searcher = Searcher(index, engine=engine)
        ranking = make_scorer(scorer)
        start = time.perf_counter()
        results = searcher.search(query, ranking, k)
        for _ in range(repeat - 1):
            results = searcher.search(query, ranking, k)
        elapsed = (time.perf_counter() - start) / repeat
        matched = searcher.matched(query)  # дерево вже в кеші
        log.info("знайдено документів: %d, пошук %.3f мс", len(matched.doc_ids), elapsed * 1000)

        terms = matched.node.terms()
        for rank, r in enumerate(results, start=1):
            snippet = None if no_snippets else snippet_for(index, r.doc_id, terms)
            rows.append(
                {
                    "rank": rank,
                    "doc_id": r.doc_id,
                    "score": round(r.score, 4),
                    "title": r.title,
                    "path": index.doc_meta[r.doc_id].path,
                    "snippet": None if snippet is None else snippet.highlight(),
                }
            )
            if snippet is None:
                highlighted.append(None)
            else:
                text = Text(snippet.text)
                for a, b in snippet.spans:
                    text.stylize("bold yellow", a, b)
                highlighted.append(text)
        found = len(matched.doc_ids)

    if as_json:
        for row in rows:
            typer.echo(json.dumps(row, ensure_ascii=False))
        return

    console = Console()
    if not rows:
        console.print(Text(f"нічого не знайдено (документів за запитом: {found})"))
        return
    console.print(Text(f"знайдено документів: {found}, показано: {len(rows)}"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("рейтинг", justify="right")
    table.add_column("назва")
    table.add_column("файл", overflow="fold")
    if not no_snippets:
        table.add_column("уривок")
    for row, snippet_text in zip(rows, highlighted, strict=True):
        cells = [
            Text(str(row["rank"])),
            Text(f"{row['score']:.3f}"),
            Text(str(row["title"])),
            Text(str(row["path"])),
        ]
        if not no_snippets:
            cells.append(snippet_text if snippet_text is not None else Text(""))
        table.add_row(*cells)
    console.print(table)


# --- findex stats ---


def _top_table(title: str, header: str, rows: list[tuple[str, int]]) -> Table:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("терм")
    table.add_column(header, justify="right")
    for rank, (term, number) in enumerate(rows, start=1):
        table.add_row(str(rank), Text(term), str(number))
    return table


@app.command("stats")
def stats_command(
    path: Annotated[Path, typer.Argument(help="тека з .txt (корпус) або файл індексу")],
    top: Annotated[int, typer.Option("--top", min=1, help="скільки термів показати")] = 50,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="лише перші N")] = None,
    eager: Annotated[bool, typer.Option("--eager", help="корпус: версія зі списками")] = False,
    no_memory: Annotated[bool, typer.Option("--no-memory", help="без tracemalloc")] = False,
    verbose: Verbose = 0,
) -> None:
    """Статистика корпусу (якщо дано теку) або індексу (якщо дано файл)."""
    setup_console()
    setup_logging(verbose)
    console = Console()

    with expected_errors():
        if path.is_dir():
            collect = collect_stats_eager if eager else collect_stats

            def job() -> Stats:
                return collect(islice(iter_documents(path), limit), top_n=top)

            measured = measure(job, trace_memory=not no_memory)
            s = measured.result
            typer.echo(f"версія:       {'eager (списки)' if eager else 'lazy (генератори)'}")
            typer.echo(f"документів:   {s.documents}")
            typer.echo(f"токенів:      {s.tokens}")
            typer.echo(f"словник:      {s.vocabulary}")
            if measured.peak_bytes is None:
                typer.echo(f"час:          {measured.elapsed:.2f} с")
                typer.echo("пік пам'яті:  не мірялась")
            else:
                typer.echo(f"час:          {measured.elapsed:.2f} с (з tracemalloc, тому довше)")
                typer.echo(f"пік пам'яті:  {mb(measured.peak_bytes):.1f} МБ")
            console.print(_top_table(f"топ-{top} термів", "кількість", s.top_terms))
        elif path.is_file():
            index = load(path)
            typer.echo(f"документів:   {index.num_docs}")
            typer.echo(f"термів:       {len(index)}")
            typer.echo(f"постінгів:    {index.num_postings()}")
            typer.echo(f"токенів:      {sum(index.doc_lengths.values())}")
            typer.echo(f"сер. довжина: {index.avg_doc_length:.1f} токенів на документ")
            typer.echo(
                f"сховище:      {index.storage}, позиції: {'так' if index.positions else 'ні'}"
            )
            by_df = sorted(index, key=lambda t: (-index.df(t), t))[:top]
            console.print(
                _top_table(
                    f"топ-{top} термів за кількістю документів",
                    "документів",
                    [(t, index.df(t)) for t in by_df],
                )
            )
        else:
            raise FileNotFoundError(f"немає такого файлу або теки: {path}")
