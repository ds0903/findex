"""Три локальні рядки таблиці лаби 7: 1 воркер async def, 1 воркер def, 4 воркери.

    python scripts/run_load_rows.py data/index.bin data/gutenberg --out docs/bench/load.json

Для кожного рядка піднімається свіжий сервер, чекаємо /health, запускаємо scripts/load_test.py.
Обережно з пам'яттю: кожен воркер тримає свій екземпляр індексу.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIRST_PORT = 8221  # кожен рядок на своєму порту: сервер, що не помер, не підмінить наступний


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def stop_tree(server: subprocess.Popen[bytes]) -> None:
    """Вбиває сервер разом з дочірніми процесами.

    У venv на Windows python.exe це лаунчер: server.kill() вбиває лише його, а справжній
    сервер і воркери лишаються жити й тримають порт та пам'ять.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True, check=False
        )
    else:
        server.kill()
    server.wait()


def wait_health(port: int, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("сервер не відповів на /health")


def load(url: str, concurrency: int, seconds: int) -> dict[str, float]:
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "load_test.py"), url]
        + ["-c", str(concurrency), "-z", str(seconds)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-c", "--concurrency", type=int, default=50)
    parser.add_argument("-z", "--seconds", type=int, default=30)
    parser.add_argument("--query", default="q=whale&k=5")
    parser.add_argument("--only", default="", help="лише рядки, назва яких містить цей текст")
    args = parser.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(REPO / "src"))
    py = sys.executable
    findex = [py, "-m", "findex", "serve", "--index", str(args.index)]
    base = [py, "-m", "findex", "serve", "--index", str(args.index)]
    base += ["--corpus-root", str(args.corpus)]
    bench = [py, str(REPO / "scripts" / "bench_sync_async.py"), str(args.index), str(args.corpus)]
    bench += ["--serve", "--cache", "256"]  # як у справжнього сервера
    rows = [
        ("1 воркер, async def", bench, "/search_async"),
        ("1 воркер, def", base + ["--workers", "1"], "/search"),
        ("4 воркери, def", base + ["--workers", "4"], "/search"),
    ]
    results = []
    for number, (name, command, route) in enumerate(rows):
        if args.only not in name:
            continue
        port = FIRST_PORT + number
        if not port_is_free(port):
            sys.exit(f"порт {port} зайнятий: лишився сервер з попереднього прогону")
        print(f"{name}: старт на порту {port}", file=sys.stderr, flush=True)
        server = subprocess.Popen(command + ["--port", str(port)], env=env, cwd=REPO)
        try:
            wait_health(port)
            load(f"http://127.0.0.1:{port}{route}?{args.query}", 4, 3)  # прогрів кешу текстів
            row = load(
                f"http://127.0.0.1:{port}{route}?{args.query}", args.concurrency, args.seconds
            )
        finally:
            stop_tree(server)
            time.sleep(2)
        if not port_is_free(port):
            print(f"УВАГА: порт {port} досі зайнятий після зупинки", file=sys.stderr, flush=True)
        row["name"] = name
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("| ціль | запитів завершено | RPS | p50, мс | p95, мс | p99, мс | помилок |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['name']} | {r['requests']:.0f} | {r['rps']} | {r['p50_ms']:.0f} "
            f"| {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | {r['errors']} |"
        )


if __name__ == "__main__":
    main()
