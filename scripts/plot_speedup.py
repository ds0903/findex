"""Таблиця й графік прискорення за результатами bench_parallel.py.

Запуск: python scripts/plot_speedup.py docs/bench/results.json docs/speedup.png
Прискорення = час serial на звичайному 3.13 (з GIL) / час конфігурації.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# відрізняються і кольором, і маркером (читається й без кольору)
STYLE = {
    ("3.13 (GIL)", "processes"): ("processes, 3.13 (GIL)", "#0072B2", "o"),
    ("3.13 (GIL)", "threads"): ("threads, 3.13 (GIL)", "#D55E00", "s"),
    ("3.13t (без GIL)", "threads"): ("threads, 3.13t (без GIL)", "#009E73", "^"),
}


def main() -> None:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    results = data["results"]
    cores = data["cores"]
    base = next(
        r["wall"] for r in results if r["python"] == "3.13 (GIL)" and r["executor"] == "serial"
    )

    print(
        "| інтерпретатор | executor | воркерів | wall, с | CPU, с "
        "| RSS парента, МБ | злиття, с | прискорення |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        speedup = base / r["wall"]
        print(
            f"| {r['python']} | {r['executor']} | {r['workers']} "
            f"| {r['wall']:.2f} | {r['cpu']:.1f} "
            f"| {r['rss_mb']:.0f} | {r['merge']:.2f} | x{speedup:.2f} |"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    xs = sorted({r["workers"] for r in results})
    ax.plot([1, cores], [1, cores], linestyle="--", color="#777777", label="ідеальне лінійне")
    for key, (label, color, marker) in STYLE.items():
        pts = sorted(
            (r["workers"], base / r["wall"]) for r in results if (r["python"], r["executor"]) == key
        )
        if pts:
            ax.plot(*zip(*pts, strict=True), marker=marker, color=color, label=label, linewidth=2)
    ax.set_xticks(xs)
    ax.set_xlabel("воркерів")
    ax.set_ylabel("прискорення відносно serial (3.13, GIL)")
    ax.set_title(f"Побудова індексу, 59 книг, {cores} логічних ядер")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(sys.argv[2])
    print(f"\nграфік: {sys.argv[2]}", file=sys.stderr)


if __name__ == "__main__":
    main()
