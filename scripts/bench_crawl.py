"""Бенчмарк краулера (лаба 6): той самий старт і max_pages, різна кількість одночасних запитів.

    python scripts/bench_crawl.py data/gutenberg --out docs/bench/crawl.json [--repeats 3]

Для кожної конфігурації й кожного повтору піднімається свіжий локальний сайт
(scripts/serve_site.py, затримка відповіді 80 мс) і запускається справжня `findex crawl`
в окремому процесі. Берется медіана wall time за повторами.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_PORT = 8800

# (назва, concurrency, per_host, delay між запитами до хоста, debug)
CONFIGS = [
    ("1 запит", 1, 1, 0.0, False),
    ("5 одночасно", 5, 5, 0.0, False),
    ("20 одночасно", 20, 20, 0.0, False),
    ("20, але 2 на хост", 20, 2, 0.0, False),
    ("20, 2 на хост, пауза 0.1 с", 20, 2, 0.1, False),
    ("20 одночасно, asyncio debug", 20, 20, 0.0, True),
]


def one_run(corpus: Path, port: int, max_pages: int, config: tuple, workdir: Path) -> dict:
    _, concurrency, per_host, delay, debug = config
    server = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "scripts" / "serve_site.py"),
            str(corpus),
            "--port",
            str(port),
            "--faults",
            "light",
        ],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        assert server.stdout is not None
        server.stdout.readline()  # "сайт на ...": сервер готовий
        stats_file = workdir / "stats.json"
        command = [
            sys.executable,
            "-m",
            "findex",
            "crawl",
            f"http://127.0.0.1:{port}/",
            "--out",
            str(workdir / "pages.jsonl"),
            "--log",
            str(workdir / "crawl.log"),
            "--stats-json",
            str(stats_file),
            "--max-pages",
            str(max_pages),
            "--concurrency",
            str(concurrency),
            "--per-host",
            str(per_host),
            "--delay",
            str(delay),
        ]
        if debug:
            command.append("--debug")
        subprocess.run(command, check=True, capture_output=True, cwd=REPO, env=_env())
        result = json.loads(stats_file.read_text(encoding="utf-8"))
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/__stats") as response:
            result["server"] = json.loads(response.read())
        return result
    finally:
        server.kill()
        server.wait()


def _env() -> dict[str, str]:
    import os

    return dict(os.environ, PYTHONIOENCODING="utf-8", COLUMNS="200")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-pages", type=int, default=300)
    args = parser.parse_args()

    rows = []
    port = BASE_PORT
    for config in CONFIGS:
        runs = []
        for k in range(args.repeats):
            print(f"{config[0]}: запуск {k + 1}/{args.repeats}", file=sys.stderr, flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                runs.append(one_run(args.corpus, port, args.max_pages, config, Path(tmp)))
            port += 1
            time.sleep(0.5)
        runs.sort(key=lambda r: r["seconds"])
        median = runs[len(runs) // 2]
        row = {
            "name": config[0],
            "concurrency": config[1],
            "per_host": config[2],
            "delay": config[3],
            "debug": config[4],
            "seconds": median["seconds"],
            "all_seconds": [r["seconds"] for r in runs],
            "pages": median["pages"],
            "errors": median["errors"],
            "retries": median["retries"],
            "pages_per_second": median["pages_per_second"],
            "peak_rss_mb": statistics.median(r["peak_rss_mb"] for r in runs),
            "server_max_active": median["server"]["max_active"],
            "slow_callbacks": median["slow_callbacks"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(
        "| конфігурація | wall, с | сторінок | сторінок/с | помилок | повторів "
        "| пік RSS, МБ | на сервері одночасно | повільних колбеків |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        slow = "" if r["slow_callbacks"] is None else r["slow_callbacks"]
        print(
            f"| {r['name']} | {r['seconds']:.2f} | {r['pages']} | {r['pages_per_second']:.1f} "
            f"| {r['errors']} | {r['retries']} | {r['peak_rss_mb']:.0f} "
            f"| {r['server_max_active']} | {slow} |"
        )


if __name__ == "__main__":
    main()
