"""Лаба 8: python проти numpy для рейтингу, memmap і стиснення різницями.

    python scripts/bench_numpy.py data/lab5_par.bin data/gutenberg --out docs/bench/numpy.json

Три частини:
  1. час пошуку (без сніпетів, кеш запитів вимкнено) на реальному корпусі й на синтетичному
     індексі з --docs документів; обидва бекенди мусять дати однакову відповідь;
  2. memmap: час відкриття індексу в Python-форматі лаби 2, у .npy повністю й у .npy через memmap;
  3. стиснення: розмір doc_id як int32 і як різниці в uint16 (на індексі, де різниці влазять).
"""

import argparse
import gc
import json
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from findex.documents import Document
from findex.index import Index, build_index
from findex.ranking import BM25
from findex.searcher import Searcher
from findex.store import load, save
from findex.vectorized import NumpyIndex, gap_decode, gap_encode


def median_ms(fn: Callable[[], object], runs: int = 7) -> float:
    started = time.perf_counter()
    fn()
    once = time.perf_counter() - started
    loops = max(1, int(0.05 / max(once, 1e-6)))
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter() - started) / loops * 1000)
    return statistics.median(samples)


def synthetic_documents(
    count: int, tokens: int = 60, vocabulary: int = 20000
) -> Iterator[Document]:
    """Слова w1..wV з розподілом Ципфа, як у справжніх текстах: мало частих, багато рідких."""
    rng = np.random.default_rng(8)
    for i in range(count):
        ranks = np.minimum(rng.zipf(1.3, tokens), vocabulary)
        yield Document(f"s{i}", Path(f"s{i}.txt"), " ".join(f"w{r}" for r in ranks), f"doc {i}")


def pick_queries(index: Index) -> list[tuple[str, str]]:
    # index.postings[term] у сховищі "array" це пара масивів, тож довжину беремо з doc_ids
    df = {term: len(index.doc_ids(term)) for term in index}
    common = max(df, key=lambda term: df[term])
    middle = min(df, key=lambda term: abs(df[term] - index.num_docs // 100))
    rare = min(df, key=lambda term: abs(df[term] - 20))
    return [
        (f"частий ({common})", common),
        (f"рідкий ({rare})", rare),
        ("3 терми через OR", f"{common} OR {middle} OR {rare}"),
    ]


def speedup(index: Index, queries: list[tuple[str, str]]) -> dict[str, object]:
    started = time.perf_counter()
    fast = Searcher(index, cache_size=0, backend="numpy")
    build_s = time.perf_counter() - started
    slow = Searcher(index, cache_size=0, backend="python")
    scorer = BM25()
    rows = []
    for label, query in queries:
        a = [(r.doc_id, r.score) for r in slow.search(query, scorer, 10)]
        b = [(r.doc_id, r.score) for r in fast.search(query, scorer, 10)]
        assert a == b, f"бекенди розійшлися на {query!r}"
        py_ms = median_ms(lambda q=query: slow.search(q, scorer, 10))
        np_ms = median_ms(lambda q=query: fast.search(q, scorer, 10))
        matched = len(slow.matched(query).doc_ids)
        rows.append(
            {"query": label, "matched": matched, "python_ms": py_ms, "numpy_ms": np_ms,
             "speedup": py_ms / np_ms}
        )  # fmt: skip
    return {
        "docs": index.num_docs,
        "postings": index.num_postings(),
        "numpy_build_s": build_s,
        "rows": rows,
    }


def touch_biggest_list(npy: Path) -> tuple[str, float]:
    """Відкриває індекс через memmap і читає найбільший список постінгів: (терм, мс)."""
    mapped = NumpyIndex.load(npy, mmap=True)
    term = max(mapped.offsets, key=mapped.df)
    return term, median_ms(lambda: mapped.postings(term)[0].sum(), runs=3)


def storage_part(docs: int) -> dict[str, object]:
    """memmap і стиснення на індексі, де всі різниці doc_id влазять у uint16 (docs <= 65536)."""
    # 200 токенів на документ: ≈4.4 млн постінгів, у 7.5 разу більше за реальний корпус
    index = build_index(synthetic_documents(docs, tokens=200), storage="array")
    npi = NumpyIndex.from_index(index)
    with tempfile.TemporaryDirectory() as tmp:
        legacy = Path(tmp) / "index.bin"
        save(index, legacy)
        npy = Path(tmp) / "npy"
        npi.save(npy)
        legacy_s = median_ms(lambda: load(legacy), runs=3) / 1000
        full_s = median_ms(lambda: NumpyIndex.load(npy, mmap=False), runs=3) / 1000
        mapped_s = median_ms(lambda: NumpyIndex.load(npy, mmap=True), runs=3) / 1000

        term, first_query_ms = touch_biggest_list(npy)
        files = {p.name: p.stat().st_size for p in npy.iterdir()}
        legacy_bytes = legacy.stat().st_size
        gc.collect()  # на Windows файл, відкритий через memmap, не можна видалити

    gaps, firsts = gap_encode(npi.doc_ids, npi.offsets)
    start, end = npi.offsets[term]
    assert list(gap_decode(gaps, int(firsts[list(npi.offsets).index(term)]), start, end)) == list(
        npi.doc_ids[start:end]
    )
    decode_ms = median_ms(lambda: gap_decode(gaps, 0, start, end))
    return {
        "docs": docs,
        "postings": int(len(npi.doc_ids)),
        "load_lab2_s": legacy_s,
        "load_npy_full_s": full_s,
        "load_npy_memmap_s": mapped_s,
        "touch_biggest_list_ms": first_query_ms,
        "lab2_file_bytes": legacy_bytes,
        "npy_files_bytes": files,
        "doc_ids_int32_bytes": int(npi.doc_ids.nbytes),
        "doc_ids_gaps_uint16_bytes": int(gaps.nbytes + firsts.nbytes),
        "decode_biggest_list_ms": decode_ms,
        "biggest_list_len": end - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="індекс лаби 2+ на реальному корпусі")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--docs", type=int, default=200_000, help="документів у синтетичному індексі"
    )
    parser.add_argument("--storage-docs", type=int, default=60_000)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    result: dict[str, object] = {}
    real = load(args.index)
    result["real"] = speedup(
        real,
        [
            ("частий (the)", "the"),
            ("рідкий (harpoon)", "harpoon"),
            ("3 терми через OR", "whale OR ship OR sea"),
        ],
    )
    del real
    if not args.skip_synthetic:
        built = time.perf_counter()
        index = build_index(synthetic_documents(args.docs), storage="array")
        print(
            f"синтетичний індекс: {args.docs} док., {time.perf_counter() - built:.0f} с", flush=True
        )
        result["synthetic"] = speedup(index, pick_queries(index))
        save_result(result, args.out)  # синтетичний індекс будується довго: результат не губимо
        del index
        result["storage"] = storage_part(args.storage_docs)
    save_result(result, args.out)

    for name in ("real", "synthetic"):
        part = result.get(name)
        if not isinstance(part, dict):
            continue
        print(f"\n{name}: {part['docs']} док., {part['postings']} постінгів")
        print(f"масиви NumPy будуються {part['numpy_build_s']:.2f} с")
        print("| запит | знайдено | python, мс | numpy, мс | прискорення |")
        print("|---|---:|---:|---:|---:|")
        for r in part["rows"]:
            cells = [r["query"], r["matched"], f"{r['python_ms']:.3f}", f"{r['numpy_ms']:.3f}"]
            print("| " + " | ".join(str(c) for c in cells) + f" | {r['speedup']:.1f}x |")
    if "storage" in result:
        print(json.dumps(result["storage"], ensure_ascii=False, indent=1))


def save_result(result: dict[str, object], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
