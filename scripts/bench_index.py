"""Заміри для лаби 2: пам'ять трьох сховищ постінгів, збереження/завантаження, ціна позицій.

Запуск: python -u scripts/bench_index.py data/gutenberg
Довго (кілька хвилин): кожне сховище будується двічі - з tracemalloc і без нього.

Збережені індекси лишаються в data/bench/. Час завантаження з них міряє bench_load.py
(у свіжому процесі), бо тут, поруч з живим оригіналом, load вийшов би занадто повільним.
"""

import gc
import sys
import time
import tracemalloc
from itertools import islice
from pathlib import Path

from findex.documents import iter_documents
from findex.index import STORAGES, PlainPosting, Posting, build_index
from findex.measure import mb, setup_console
from findex.store import load, save

POSITIONS_LIMIT = 20  # позиції міряємо на перших 20 документах, бо на всьому корпусі їх забагато


def docs(root: Path, limit: int | None = None):
    return islice(iter_documents(root), limit)


def traced_build(root, storage, limit=None, positions=False):
    """Повертає (індекс, скільки пам'яті лишилось після побудови, пік під час побудови)."""
    gc.collect()
    tracemalloc.start()
    index = build_index(docs(root, limit), storage, positions)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return index, current, peak


def timed(func, *args):
    start = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - start


def per_instance(cls, n=1_000_000):
    """Скільки байтів займає один екземпляр (без вказівника на нього зі списку)."""
    gc.collect()
    tracemalloc.start()
    items = [cls(i % 59, 3) for i in range(n)]
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (current - sys.getsizeof(items)) / n


def main() -> None:
    setup_console()
    root = Path(sys.argv[1])
    tmp = Path("data/bench")
    tmp.mkdir(parents=True, exist_ok=True)

    print("## пам'ять і файли: три сховища постінгів\n", flush=True)
    rows = []
    save_rows = []
    parts = {}
    n_post = n_terms = 0
    for storage in STORAGES:
        print(f"[{storage}] build з tracemalloc...", file=sys.stderr, flush=True)
        index, retained, peak = traced_build(root, storage)
        n_post, n_terms = index.num_postings(), len(index.postings)
        del index
        gc.collect()

        print(f"[{storage}] build без tracemalloc, save/load...", file=sys.stderr, flush=True)
        index, build_time = timed(build_index, docs(root), storage)
        fmt_info = {}
        for fmt in ("pickle", "binary"):
            path = tmp / f"{storage}.{fmt}"
            _, t_save = timed(save, index, path, fmt)
            loaded = load(path)  # час тут не міряємо, лише перевіряємо, що все збереглося
            assert loaded == index, f"{storage}/{fmt}: після load індекс не такий самий"
            del loaded
            gc.collect()
            size = path.stat().st_size
            fmt_info[fmt] = (size, t_save)
            save_rows.append(f"| {storage} | {fmt} | {mb(size):.1f} | {t_save:.2f} |")

        rows.append(
            f"| {storage} | {mb(peak):.0f} | {mb(retained):.0f} | {retained / n_post:.0f} "
            f"| {mb(fmt_info['pickle'][0]):.1f} | {build_time:.1f} |"
        )

        # складові, щоб пояснити, куди пішли байти (sys.getsizeof рахує сам об'єкт, не вкладене)
        base = sys.getsizeof(index.postings) + sum(sys.getsizeof(t) for t in index.postings)
        entries = list(index.postings.values())
        if storage == "array":
            containers = sum(sys.getsizeof(e) for e in entries)  # кортежі (пари)
            arrays = sum(sys.getsizeof(a) for e in entries for a in e)
            parts[storage] = (n_terms, n_post, base, containers, arrays)
        else:
            lists = sum(sys.getsizeof(e) for e in entries)
            first = entries[0][0]
            big_tf = sum(1 for t in index for p in index.iter_postings(t) if p.tf > 256)
            parts[storage] = (n_terms, n_post, base, lists, sys.getsizeof(first), big_tf)
        del index, entries
        gc.collect()

    print(f"постінгів: {n_post}, термів: {n_terms}\n")
    print(
        "| сховище | пік побудови, МБ | лишилось після побудови, МБ | байтів на постінг "
        "| pickle, МБ | побудова без tracemalloc, с |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    print("\n".join(rows))

    print("\n## збереження (час без tracemalloc)\n")
    print("| сховище | формат | розмір, МБ | save, с |")
    print("|---|---|---:|---:|")
    print("\n".join(save_rows))

    print("\n## складові (sys.getsizeof)\n")
    for storage, p in parts.items():
        if storage == "array":
            n_terms, n_post, base, containers, arrays = p
            print(
                f"- {storage}: термів {n_terms}; словник + рядки термів {mb(base):.1f} МБ; "
                f"кортежі-пари {mb(containers):.1f} МБ; усі масиви {mb(arrays):.1f} МБ, "
                f"з них корисних даних {mb(n_post * 8):.1f} МБ (постінгів {n_post} x 2 x 4 Б)"
            )
        else:
            n_terms, n_post, base, lists, inst, big_tf = p
            print(
                f"- {storage}: словник + рядки термів {mb(base):.1f} МБ; списки (вказівники) "
                f"{mb(lists):.1f} МБ; getsizeof одного екземпляра {inst} Б "
                f"(x {n_post} = {mb(inst * n_post):.1f} МБ); постінгів з tf > 256 "
                f"(окремий int 28 Б): {big_tf}"
            )

    print("\n## один екземпляр, 1 000 000 штук (tracemalloc)\n")
    print("| клас | байтів на екземпляр |")
    print("|---|---:|")
    print(f"| PlainPosting (без slots) | {per_instance(PlainPosting):.1f} |")
    print(f"| Posting (slots=True) | {per_instance(Posting):.1f} |")
    print("| елемент array('I') | 4.0 |")

    print(f"\n## ціна позицій (перші {POSITIONS_LIMIT} документів, сховище slots)\n", flush=True)
    print("| позиції | постінгів | пік побудови, МБ | лишилось, МБ | pickle, МБ | binary, МБ |")
    print("|---|---:|---:|---:|---:|---:|")
    for positions in (False, True):
        print(f"[positions={positions}]...", file=sys.stderr, flush=True)
        index, retained, peak = traced_build(root, "slots", POSITIONS_LIMIT, positions)
        sizes = []
        for fmt in ("pickle", "binary"):
            path = tmp / f"pos-{positions}.{fmt}"
            save(index, path, fmt)
            sizes.append(mb(path.stat().st_size))
        print(
            f"| {'так' if positions else 'ні'} | {index.num_postings()} | {mb(peak):.0f} "
            f"| {mb(retained):.0f} | {sizes[0]:.1f} | {sizes[1]:.1f} |",
            flush=True,
        )
        del index
        gc.collect()


if __name__ == "__main__":
    main()
