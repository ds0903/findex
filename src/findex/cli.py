"""Командний рядок findex: `findex index`, `findex search`, `findex stats`.

Правила для всіх команд:
- результат (таблиця, JSON, підсумок) іде в stdout, усе діагностичне (логи, прогрес) у stderr;
- логування налаштовується тут і лише тут (`-v` це INFO, `-vv` це DEBUG), бібліотечний код
  тільки пише в свої логери;
- очікувані помилки (немає файлу, поганий запит, пошкоджений індекс) це один рядок у stderr
  і ненульовий код виходу, без traceback.
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Generator
from contextlib import aclosing, contextmanager
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.live import Live
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

from findex.crawl import CrawlConfig, CrawlStats, FetchConfig, crawl, default_user_agent
from findex.documents import find_paths, iter_documents
from findex.index import Storage, build_index
from findex.logconfig import setup_logging
from findex.measure import mb, measure, peak_rss_mb, setup_console
from findex.parallel import BuildResult, ExecutorName, build_parallel
from findex.ranking import ScorerName, make_scorer
from findex.search import Engine
from findex.searcher import Searcher
from findex.snippets import snippet_for
from findex.stats import Stats, collect_stats, collect_stats_eager
from findex.store import Format, load, open_index, save

log = logging.getLogger(__name__)

# менше за цей розмір корпусу пул процесів не окупається (запуск воркерів ≈ їхня робота)
AUTO_PARALLEL_BYTES = 5 * 1024 * 1024

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
    root: Annotated[Path, typer.Argument(help="тека з .txt файлами або .jsonl від `findex crawl`")],
    out: Annotated[Path, typer.Option("--out", "-o", help="куди зберегти індекс")],
    fmt: Annotated[Format, typer.Option("--format", help="формат файлу")] = "binary",
    storage: Annotated[Storage, typer.Option("--storage", help="як тримати постінги")] = "slots",
    positions: Annotated[bool, typer.Option("--positions", help="зберігати позиції")] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="лише перші N")] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-w", min=1, help="кількість воркерів (за замовч. авто)"),
    ] = None,
    executor: Annotated[
        ExecutorName, typer.Option("--executor", help="чим розпаралелювати")
    ] = "processes",
    no_memory: Annotated[bool, typer.Option("--no-memory", help="без tracemalloc")] = False,
    verbose: Verbose = 0,
) -> None:
    """Будує індекс з усіх .txt у теці й зберігає його у файл.

    Воркери: за замовчуванням авто. Для корпусу менше 5 МБ це 1 (пул не окупається),
    інакше стільки, скільки ядер. З --workers 1 пул не створюється взагалі.
    Файл .jsonl (результат `findex crawl`) індексується в один потік: його не розбити на шматки.
    """
    setup_console()
    setup_logging(verbose)

    with expected_errors():
        root = root.resolve()
        from_crawl = root.suffix == ".jsonl" and root.is_file()
        paths = [] if from_crawl else find_paths(root)
        if not paths and not from_crawl:
            raise ValueError(f"у {root} немає .txt файлів")
        paths = paths[:limit]
        if workers is None:
            big = sum(p.stat().st_size for p in paths) > AUTO_PARALLEL_BYTES
            workers = min(os.cpu_count() or 1, len(paths)) if big else 1
        used = "serial" if workers == 1 or from_crawl else executor
        workers = 1 if from_crawl else workers
        log.info("файлів до індексації: %d, воркерів: %d, executor: %s", len(paths), workers, used)

        def job() -> tuple[BuildResult, float]:
            with _progress() as progress:
                task = progress.add_task("індексація", total=len(paths) or None)
                if from_crawl:
                    began = time.perf_counter()
                    pages = islice(iter_documents(root), limit)
                    built = build_index(pages, storage, positions, root=str(root))
                    result = BuildResult(built, time.perf_counter() - began, 0.0, 0.0)
                else:
                    result = build_parallel(
                        root,
                        paths,
                        workers,
                        used,
                        storage,
                        positions,
                        on_chunk_done=lambda n: progress.advance(task, n),
                    )
            start = time.perf_counter()
            save(result.index, out, fmt)
            return result, time.perf_counter() - start

        measured = measure(job, trace_memory=not no_memory)

    built, saved = measured.result
    index = built.index
    log.info("індекс збережено в %s", out)
    typer.echo(f"документів:   {index.num_docs}")
    typer.echo(f"термів:       {len(index)}")
    typer.echo(f"постінгів:    {index.num_postings()}")
    typer.echo(f"сховище:      {index.storage}, позиції: {'так' if index.positions else 'ні'}")
    typer.echo(f"воркери:      {workers}, executor: {used}")
    typer.echo(f"побудова:     {built.build_seconds + built.merge_seconds:.2f} с")
    typer.echo(f"  з них злиття: {built.merge_seconds:.2f} с")
    typer.echo(f"збереження:   {saved:.2f} с -> {out} ({mb(out.stat().st_size):.1f} МБ, {fmt})")
    if measured.peak_bytes is None:
        typer.echo("пік пам'яті:  не мірялась")
    else:
        typer.echo(
            f"пік пам'яті:  {mb(measured.peak_bytes):.1f} МБ (tracemalloc, тому час більший)"
        )


# --- findex crawl ---

# у тестах сюди підставляють MockTransport; у роботі лишається None, тобто справжня мережа
_TRANSPORT: httpx.AsyncBaseTransport | None = None


class _SlowCallbackCounter(logging.Handler):
    """Режим debug в asyncio пише попередження, якщо задача надовго зайняла цикл подій."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if " took " in message:  # "Executing <Task ...> took 0.153 seconds"
            self.messages.append(message)


def _crawl_view(stats: CrawlStats, max_pages: int) -> Table:
    table = Table(show_header=True, header_style="bold")
    for name in ("зібрано", "стор/с", "чекають відповіді", "помилок", "повторів", "у черзі", "МБ"):
        table.add_column(name, justify="right")
    table.add_row(
        f"{stats.pages}/{max_pages}",
        f"{stats.pages_per_second:.1f}",
        str(stats.in_flight),
        str(stats.errors),
        str(stats.retries),
        str(stats.queued),
        f"{mb(stats.nbytes):.1f}",
    )
    return table


async def _crawl_to_file(
    urls: list[str], config: CrawlConfig, out: Path, stats: CrawlStats
) -> None:
    with out.open("w", encoding="utf-8") as sink:
        # aclosing: при помилці запису генератор закриється, воркери скасуються
        async with aclosing(crawl(urls, config, stats=stats, transport=_TRANSPORT)) as pages:
            async for page in pages:
                sink.write(page.to_json() + "\n")  # сторінка пишеться й забувається
                sink.flush()


@app.command("crawl")
def crawl_command(
    urls: Annotated[list[str], typer.Argument(help="початкові адреси (http/https)")],
    out: Annotated[Path, typer.Option("--out", "-o", help="куди писати JSON lines")] = Path(
        "pages.jsonl"
    ),
    max_pages: Annotated[int, typer.Option("--max-pages", min=1, help="стеля на адреси")] = 100,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="запитів одночасно")
    ] = 5,
    per_host: Annotated[int, typer.Option("--per-host", min=1, help="з них до одного хоста")] = 2,
    delay: Annotated[
        float, typer.Option("--delay", min=0, help="пауза між запитами до хоста, с")
    ] = 0.5,
    max_depth: Annotated[int, typer.Option("--max-depth", min=0, help="глибина від старту")] = 10,
    allow_host: Annotated[
        list[str] | None,
        typer.Option("--allow-host", help="ще один дозволений хост (host[:port]); можна кілька"),
    ] = None,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1, help="секунд на запит")] = 10.0,
    user_agent: Annotated[
        str | None, typer.Option("--user-agent", help="за замовч. findex-crawler/версія (+адреса)")
    ] = None,
    log_file: Annotated[Path, typer.Option("--log", help="статус кожного URL")] = Path("crawl.log"),
    stats_json: Annotated[
        Path | None, typer.Option("--stats-json", help="підсумок у JSON (для бенчмарків)")
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="asyncio debug: показати, чи хтось блокував цикл")
    ] = False,
    verbose: Verbose = 0,
) -> None:
    """Обходить сайт і пише сторінки у JSON lines. Поважає robots.txt і не виходить за хости.

    Обходьте лише те, що вам дозволено: свій сайт, документацію з відкритим robots.txt,
    локальне дзеркало. Результат читає `findex index pages.jsonl`.
    """
    setup_console()
    setup_logging(verbose)

    with expected_errors():
        config = CrawlConfig(
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_delay=delay,
            max_depth=max_depth,
            allowed_hosts=frozenset(allow_host or []),
            user_agent=user_agent or default_user_agent(),
            fetch=FetchConfig(timeout=timeout),
        )
        stats = CrawlStats()
        slow = _SlowCallbackCounter()

        # статус кожного URL і всі повтори йдуть у crawl.log; на екран лише за -v
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        watched = [logging.getLogger("findex.crawl"), logging.getLogger("asyncio")]
        saved = [(lg, lg.level, lg.propagate) for lg in watched]
        for lg in watched:
            lg.addHandler(file_handler)
            lg.setLevel(min(lg.getEffectiveLevel(), logging.INFO))
            lg.propagate = verbose > 0  # інакше рядки логу псували б живу таблицю
        logging.getLogger("asyncio").addHandler(slow)
        try:
            console = Console(stderr=True)
            view = Live(
                get_renderable=lambda: _crawl_view(stats, max_pages),
                console=console,
                refresh_per_second=4,
            )
            with view:
                asyncio.run(_crawl_to_file(urls, config, out, stats), debug=debug)
        finally:
            for lg, level, propagate in saved:
                lg.removeHandler(file_handler)
                lg.setLevel(level)
                lg.propagate = propagate
            logging.getLogger("asyncio").removeHandler(slow)
            file_handler.close()

    elapsed = stats.elapsed
    summary: dict[str, object] = {
        "pages": stats.pages,
        "errors": stats.errors,
        "retries": stats.retries,
        "robots_blocked": stats.robots_blocked,
        "bytes": stats.nbytes,
        "seconds": round(elapsed, 3),
        "pages_per_second": round(stats.pages / elapsed, 2) if elapsed > 0 else 0.0,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "slow_callbacks": len(slow.messages) if debug else None,
    }
    if stats_json is not None:
        stats_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    typer.echo(f"сторінок:     {stats.pages} -> {out}")
    typer.echo(f"помилок:      {stats.errors}, повторів: {stats.retries}")
    typer.echo(f"robots.txt:   пропущено адрес: {stats.robots_blocked}")
    typer.echo(f"завантажено:  {mb(stats.nbytes):.1f} МБ за {elapsed:.2f} с")
    typer.echo(f"швидкість:    {summary['pages_per_second']} сторінок/с")
    typer.echo(f"пік пам'яті:  {summary['peak_rss_mb']} МБ (RSS)")
    typer.echo(f"журнал:       {log_file}")
    if debug:
        typer.echo(f"asyncio debug: повільних колбеків (>100 мс): {len(slow.messages)}")
        for message in slow.messages[:5]:
            typer.echo(f"  {message}", err=True)


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
