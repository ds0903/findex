"""Паралельна побудова індексу: build_partial для шматка файлів + merge.

Схема: список файлів ріжеться на послідовні шматки (chunks) приблизно однакового розміру
в байтах; кожному шматку наперед дається діапазон doc_id (first_doc_id). Воркер отримує
лише шляхи файлів (не тексти), сам читає, токенізує й будує частковий індекс. Діапазони
не перетинаються, тому merge просто дописує списки постінгів термів у порядку шматків,
і вони лишаються відсортованими за doc_id без жодного пересортування.

Окремих блокувань немає й бути не повинно: воркери нічого не діляться, кожен будує свій
Index, а об'єднує їх лише головний потік. Поділ одного Counter між потоками дає втрату
оновлень, див. scripts/race_demo.py.

Функції build_partial і merge лежать на рівні модуля, бо ProcessPoolExecutor передає
дочірньому процесу функцію за ім'ям (pickle), а локальні функції й lambda так не передаються.
"""

import concurrent.futures as cf
import logging
import multiprocessing
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from findex.documents import Document, read_document
from findex.index import Index, Storage, build_index
from findex.timing import timed

log = logging.getLogger(__name__)

ExecutorName = Literal["serial", "threads", "processes"]
EXECUTOR_NAMES: tuple[ExecutorName, ...] = ("serial", "threads", "processes")


@dataclass(frozen=True)
class Chunk:
    first_doc_id: int
    paths: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    index: Index
    build_seconds: float  # від початку до кінця роботи воркерів
    merge_seconds: float
    worker_cpu_seconds: float  # CPU воркерів (для processes), 0 для serial і threads


def plan_chunks(paths: Sequence[Path | str], n_chunks: int, sizes: Sequence[int]) -> list[Chunk]:
    """Ріже paths на не більше n_chunks послідовних шматків з близькою сумою розмірів.

    Шматки послідовні (не перемішані), щоб doc_id у них були неперервними діапазонами
    і списки постінгів при злитті трималися відсортованими.
    """
    if len(sizes) != len(paths):
        raise ValueError("sizes і paths мають бути однакової довжини")
    if n_chunks < 1:
        raise ValueError("n_chunks має бути не менше 1")
    n_chunks = min(n_chunks, len(paths))
    if n_chunks == 0:
        return []

    target = sum(sizes) / n_chunks
    chunks: list[Chunk] = []
    current: list[str] = []
    current_size = 0
    first = 0
    for i, (path, size) in enumerate(zip(paths, sizes, strict=True)):
        current.append(str(path))
        current_size += size
        left_files = len(paths) - i - 1
        left_chunks = n_chunks - len(chunks) - 1
        # закриваємо шматок, коли набрали свою частку, але лишаємо по файлу на кожен наступний
        if left_chunks > 0 and (current_size >= target or left_files == left_chunks):
            chunks.append(Chunk(first, tuple(current)))
            first += len(current)
            current, current_size = [], 0
    if current:
        chunks.append(Chunk(first, tuple(current)))
    return chunks


def _read_all(root: Path, paths: Sequence[str]) -> Iterator[Document]:
    for p in paths:
        try:
            yield read_document(Path(p), root)
        except (OSError, UnicodeDecodeError) as exc:
            # у паралельній побудові діапазон id вже роздано, тому пропустити файл не можна:
            # id пішли б з дірками, а Index.all_doc_ids() це range(num_docs)
            raise ValueError(f"не можу прочитати {p}: {exc}") from exc


def build_partial(
    root: str,
    paths: Sequence[str],
    first_doc_id: int,
    storage: Storage = "slots",
    positions: bool = False,
) -> Index:
    """Індекс для одного шматка файлів, doc_id від first_doc_id."""
    return build_index(
        _read_all(Path(root), paths), storage, positions, root=root, first_doc_id=first_doc_id
    )


def _run_chunk(root: str, chunk: Chunk, storage: Storage, positions: bool) -> tuple[Index, float]:
    """Те, що виконує воркер: частковий індекс і скільки CPU на нього пішло."""
    start = time.thread_time()
    index = build_partial(root, chunk.paths, chunk.first_doc_id, storage, positions)
    return index, time.thread_time() - start


@timed
def merge(partials: Sequence[Index]) -> Index:
    """Зливає часткові індекси в один. Часткові індекси при цьому "споживаються":
    їхні списки постінгів дописуються, а не копіюються.

    partials мають іти в порядку зростання діапазонів doc_id.
    """
    if not partials:
        raise ValueError("нічого зливати")
    first = partials[0]
    merged = Index(
        postings={},
        doc_lengths={},
        doc_meta={},
        storage=first.storage,
        positions=first.positions,
        root=first.root,
    )
    for part in partials:
        if (part.storage, part.positions) != (first.storage, first.positions):
            raise ValueError("часткові індекси побудовані з різними параметрами")
        if not merged.doc_meta.keys().isdisjoint(part.doc_meta):
            raise ValueError("діапазони doc_id часткових індексів перетинаються")
        merged.doc_lengths.update(part.doc_lengths)
        merged.doc_meta.update(part.doc_meta)
        for term, entry in part.postings.items():
            existing = merged.postings.get(term)
            if existing is None:
                merged.postings[term] = entry
            elif isinstance(existing, tuple) and isinstance(entry, tuple):
                existing[0].extend(entry[0])
                existing[1].extend(entry[1])
            elif isinstance(existing, list) and isinstance(entry, list):
                cast("list[Any]", existing).extend(entry)
            else:
                raise ValueError(f"терм {term!r}: різні сховища в часткових індексах")
    return merged


def _make_executor(name: ExecutorName, workers: int) -> cf.Executor | None:
    if name == "serial":
        return None
    if name == "threads":
        return cf.ThreadPoolExecutor(max_workers=workers)
    # spawn однаково працює на windows, linux і macos (fork небезпечний разом з потоками)
    return cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )


def build_parallel(
    root: Path | str,
    paths: Sequence[Path],
    workers: int = 1,
    executor: ExecutorName = "processes",
    storage: Storage = "slots",
    positions: bool = False,
    on_chunk_done: Callable[[int], None] | None = None,
) -> BuildResult:
    """Будує індекс по файлах paths. on_chunk_done(n) викликається, коли готовий шматок з n файлів.

    serial: усе в поточному потоці, без пулу (еталон для порівняння).
    Якщо воркер впав, виняток піднімається тут, решта шматків скасовується, пул закривається:
    зависання нема.
    """
    if workers < 1:
        raise ValueError("workers має бути не менше 1")
    if executor not in EXECUTOR_NAMES:
        raise ValueError(f"невідомий executor {executor!r}, є: {', '.join(EXECUTOR_NAMES)}")
    root_str = str(root)
    if not paths:
        raise ValueError("немає файлів для індексації")

    sizes = [p.stat().st_size for p in paths]
    n_chunks = 1 if executor == "serial" else workers * 4
    chunks = plan_chunks(paths, n_chunks, sizes)
    log.info("executor=%s, workers=%d, шматків: %d", executor, workers, len(chunks))

    start = time.perf_counter()
    results: list[Index | None] = [None] * len(chunks)
    cpu = 0.0
    pool = _make_executor(executor, workers)
    try:
        if pool is None:
            for i, chunk in enumerate(chunks):
                results[i], _ = _run_chunk(root_str, chunk, storage, positions)
                if on_chunk_done:
                    on_chunk_done(len(chunk.paths))
        else:
            futures = {
                pool.submit(_run_chunk, root_str, chunk, storage, positions): i
                for i, chunk in enumerate(chunks)
            }
            for future in cf.as_completed(futures):
                i = futures[future]
                results[i], seconds = future.result()  # тут піднімається виняток воркера
                if executor == "processes":
                    cpu += seconds
                if on_chunk_done:
                    on_chunk_done(len(chunks[i].paths))
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        if pool is not None:
            pool.shutdown()
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parts = [r for r in results if r is not None]
    index = merge(parts)
    return BuildResult(index, build_seconds, time.perf_counter() - start, cpu)
