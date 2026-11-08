"""Гонка даних на спільному Counter і два способи її прибрати.

Запуск: python scripts/race_demo.py [потоків] [повторень]

Кілька потоків рахують слова в ОДИН спільний Counter. `counts[w] += 1` це не одна дія, а три:
прочитати значення, додати 1, записати назад. Якщо потік перемкнули між читанням і записом,
чужий приріст затирається ("втрачене оновлення"). Три варіанти:

1. shared  - спільний Counter без захисту (неправильно);
2. lock    - той самий Counter, кожне оновлення під Lock (правильно, але потоки чекають);
3. partial - у кожного потоку свій локальний Counter, потім злиття (правильно й без очікування;
             це та сама ідея, що build_partial + merge в findex.parallel).

На звичайному CPython (з GIL) переключення між потоками рідкісне, тому для наочності
ставимо дуже малий switchinterval. На free-threaded 3.13t гонка проявляється й без цього.
"""

import sys
import threading
import time
from collections import Counter
from collections.abc import Callable

WORDS = ["whale", "ship", "sea", "harpoon", "captain"]


def run_threads(target: Callable[[int], None], n_threads: int) -> float:
    threads = [threading.Thread(target=target, args=(i,)) for i in range(n_threads)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def shared(n_threads: int, per_thread: int) -> tuple[int, float]:
    counts: Counter[str] = Counter()

    def work(_: int) -> None:
        for i in range(per_thread):
            counts[WORDS[i % len(WORDS)]] += 1  # читання, +1, запис: не атомарно

    seconds = run_threads(work, n_threads)
    return counts.total(), seconds


def with_lock(n_threads: int, per_thread: int) -> tuple[int, float]:
    counts: Counter[str] = Counter()
    lock = threading.Lock()

    def work(_: int) -> None:
        for i in range(per_thread):
            with lock:
                counts[WORDS[i % len(WORDS)]] += 1

    seconds = run_threads(work, n_threads)
    return counts.total(), seconds


def with_partials(n_threads: int, per_thread: int) -> tuple[int, float]:
    partials: list[Counter[str]] = [Counter() for _ in range(n_threads)]

    def work(k: int) -> None:
        local = partials[k]  # цей Counter чіпає лише потік k
        for i in range(per_thread):
            local[WORDS[i % len(WORDS)]] += 1

    start = time.perf_counter()
    run_threads(work, n_threads)
    total: Counter[str] = Counter()
    for part in partials:  # злиття в головному потоці, після join
        total.update(part)
    return total.total(), time.perf_counter() - start


def main() -> None:
    n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    per_thread = 300_000
    sys.setswitchinterval(1e-6)  # див. докстринг
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    expected = n_threads * per_thread
    print(f"python {sys.version.split()[0]}, GIL увімкнено: {gil}, потоків: {n_threads}")
    print(f"очікувана сума: {expected}\n")
    print("| варіант | запуск | отримано | втрачено | час, с |")
    print("|---|---:|---:|---:|---:|")
    for name, func in (("shared", shared), ("lock", with_lock), ("partial", with_partials)):
        for run in range(1, repeats + 1):
            got, seconds = func(n_threads, per_thread)
            print(f"| {name} | {run} | {got} | {expected - got} | {seconds:.2f} |")


if __name__ == "__main__":
    main()
