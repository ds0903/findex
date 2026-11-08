"""Бенчмарк паралельної побудови індексу (лаба 5).

Запуск (усі виміри, кілька хвилин; краще на тихій машині, без інших важких програм):

    python scripts/bench_parallel.py data/gutenberg --out docs/bench/results.json \
        [--python-ft path/to/python3.13t.exe] [--repeats 3]

Кожен замір це окремий свіжий процес Python (`--one`), щоб пам'ять і кеші попереднього запуску
не впливали. Для кожної конфігурації береться медіана за --repeats запусків.
Міряється: wall (побудова + злиття), CPU (парент + воркери), пік RSS парента, час злиття.

CPU воркерів-процесів рахується в самих воркерах (time.thread_time на шматок) і не включає
запуск інтерпретатора й імпорт модулів у воркері; для потоків CPU це CPU всього процесу.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def peak_rss_mb() -> float:
    """Пік робочого набору поточного процесу, МБ."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 1024 / 1024
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / (1024 if sys.platform == "darwin" else 1)


def run_one(root: Path, executor: str, workers: int) -> dict[str, object]:
    from findex.documents import find_paths
    from findex.parallel import build_parallel

    root = root.resolve()
    paths = find_paths(root)
    cpu0 = time.process_time()
    result = build_parallel(root, paths, workers, executor)  # type: ignore[arg-type]  # рядок з CLI
    parent_cpu = time.process_time() - cpu0
    return {
        "wall": result.build_seconds + result.merge_seconds,
        "cpu": parent_cpu + result.worker_cpu_seconds,
        "merge": result.merge_seconds,
        "rss_mb": peak_rss_mb(),
        "terms": len(result.index),
        "postings": result.index.num_postings(),
    }


def run_in_fresh_process(python: str, root: Path, executor: str, workers: int) -> dict[str, object]:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONIOENCODING="utf-8")
    out = subprocess.run(
        [python, str(Path(__file__)), "--one", executor, str(workers), str(root)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def median_row(runs: list[dict[str, object]]) -> dict[str, float]:
    return {
        key: statistics.median(float(r[key]) for r in runs)  # type: ignore[arg-type]  # з JSON
        for key in ("wall", "cpu", "merge", "rss_mb")
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        _, _, executor, workers, root = sys.argv
        print(json.dumps(run_one(Path(root), executor, int(workers))))
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python-ft", help="шлях до python3.13t (без GIL)")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cores})
    interpreters = {"3.13 (GIL)": sys.executable}
    if args.python_ft:
        interpreters["3.13t (без GIL)"] = args.python_ft

    configs: list[tuple[str, str, int]] = []
    for name in interpreters:
        configs.append((name, "serial", 1))
        if name.startswith("3.13 "):
            configs += [(name, "processes", w) for w in worker_counts]
        configs += [(name, "threads", w) for w in worker_counts]

    results: list[dict[str, object]] = []
    for name, executor, workers in configs:
        runs = []
        for k in range(args.repeats):
            print(
                f"{name} {executor} x{workers} запуск {k + 1}/{args.repeats}",
                file=sys.stderr,
                flush=True,
            )
            runs.append(run_in_fresh_process(interpreters[name], args.root, executor, workers))
        row = median_row(runs)
        results.append(
            {
                "python": name,
                "executor": executor,
                "workers": workers,
                **row,
                "all_wall": [r["wall"] for r in runs],
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), file=sys.stderr, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"cores": cores, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"збережено {args.out}")


if __name__ == "__main__":
    main()
