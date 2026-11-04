"""Час завантаження індексу у свіжому процесі.

Запуск: python scripts/bench_load.py FILE [FILE ...]
Файли лишає після себе bench_index.py (тека з назвою findex-bench-* у тимчасових).

Чому окремий процес: у bench_index.py оригінал індексу ще лежить у пам'яті, коли ми
завантажуємо копію, і збирач сміття (GC) обходить обидва. Реальна програма
(`python -m findex.search`) стартує з порожньої пам'яті, тож міряємо так само.
Береться мінімум із трьох запусків.
"""

import subprocess
import sys
from pathlib import Path

from findex.measure import setup_console

CODE = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from findex.store import load\n"
    "start = time.perf_counter()\n"
    "load(Path(sys.argv[1]))\n"
    "print(time.perf_counter() - start)\n"
)


def load_time(path: Path, runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        done = subprocess.run(
            [sys.executable, "-I", "-c", CODE, str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        times.append(float(done.stdout))
    return min(times)


def main() -> None:
    setup_console()
    print("| файл | розмір, МБ | load у свіжому процесі, с |")
    print("|---|---:|---:|")
    for name in sys.argv[1:]:
        path = Path(name)
        size = path.stat().st_size / 1024 / 1024
        print(f"| {path.name} | {size:.1f} | {load_time(path):.2f} |", flush=True)


if __name__ == "__main__":
    main()
