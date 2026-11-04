import gc
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from findex.documents import Document
from findex.tokenizer import tokenize


@dataclass(frozen=True)
class Stats:
    documents: int
    tokens: int
    vocabulary: int
    top_terms: list[tuple[str, int]]


@dataclass(frozen=True)
class Measurement:
    stats: Stats
    elapsed: float  # секунди
    peak_bytes: int | None  # None, якщо tracemalloc вимкнено


def collect_stats(documents: Iterable[Document], top_n: int = 50) -> Stats:
    """Один прохід по документах. Тексти й токени не накопичуються,
    в пам'яті лише поточний документ та лічильник термів."""
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in documents:
        n_docs += 1
        counts.update(tokenize(doc.text))
    return Stats(
        documents=n_docs,
        tokens=counts.total(),
        vocabulary=len(counts),
        top_terms=counts.most_common(top_n),
    )


def collect_stats_eager(documents: Iterable[Document], top_n: int = 50) -> Stats:
    """«Жадібна» версія для порівняння: все спочатку складаємо в списки."""
    docs = list(documents)  # усі тексти одночасно в пам'яті
    all_tokens: list[str] = []
    for doc in docs:
        tokens = list(tokenize(doc.text))  # ще один список на документ
        all_tokens.extend(tokens)  # і великий загальний список
    counts = Counter(all_tokens)
    return Stats(
        documents=len(docs),
        tokens=len(all_tokens),
        vocabulary=len(counts),
        top_terms=counts.most_common(top_n),
    )


def measure(func: Callable[[], Stats], trace_memory: bool = True) -> Measurement:
    """Запускає func, міряє час (perf_counter) і піковий обсяг пам'яті (tracemalloc).

    Увага: з увімкненим tracemalloc програма працює в рази повільніше,
    тож для чесного часу запускайте з trace_memory=False.
    """
    gc.collect()
    if trace_memory:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        stats = func()
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if trace_memory else None
    finally:
        if trace_memory:
            tracemalloc.stop()
    return Measurement(stats=stats, elapsed=elapsed, peak_bytes=peak)
