"""def проти async def для /search (лаба 7): 20 одночасних запитів.

    python scripts/bench_sync_async.py data/index_pos.bin data/gutenberg \
        --out docs/bench/sync_async.json

Піднімає сервер в окремому процесі (1 воркер uvicorn) з трьома однаковими маршрутами:
  /search_def      звичайний def: FastAPI виконує його в пулі потоків;
  /search_async    async def, що рахує прямо в циклі подій (блокує його);
  /search_thread   async def, що рахує через asyncio.to_thread (пул потоків вручну).
Кеш запитів вимкнено (cache_size=0), бо інакше повторні запити не мали б роботи. Клієнт у
своєму процесі шле 20 запитів одночасно, десять раундів поспіль, і весь цей час опитує /health
кожні 5 мс: якщо цикл подій зайнятий, /health відповідає із затримкою.
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

QUERIES = [
    "whale", "ship AND sea", "love OR death", "king AND queen", "night AND day",
    "blood", "storm AND ship", "heart", "war AND peace", "ghost OR spirit",
    "white AND sea", "gold", "battle", "river AND bank", "mother AND son",
    "house", "dark AND night", "fire", "sword", "island",
]  # fmt: skip
CONCURRENCY = 20
ROUNDS = 4  # один запит це 0.4-1.3 с процесора (в основному сніпети), тож раунд триває секунд 15


def serve(index: Path, corpus: Path, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI, Request

    from findex.web import create_app
    from findex.web.service import run_search
    from findex.web.settings import Settings

    app: FastAPI = create_app(Settings(index_path=index, corpus_root=corpus, cache_size=0))

    def work(request: Request, q: str):
        state = request.app.state.findex
        return run_search(state.index, state.searcher, state.text_of, q, 10, "bm25", 1)

    @app.get("/search_def")
    def search_def(request: Request, q: str):
        return work(request, q)

    @app.get("/search_async")
    async def search_async(request: Request, q: str):
        return work(request, q)  # синхронна робота прямо в циклі подій

    @app.get("/search_thread")
    async def search_thread(request: Request, q: str):
        return await asyncio.to_thread(work, request, q)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error", access_log=False)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


async def measure(client: httpx.AsyncClient, route: str) -> dict[str, float]:
    latencies: list[float] = []
    walls: list[float] = []
    health: list[float] = []
    stop = asyncio.Event()

    async def one(q: str) -> None:
        started = time.perf_counter()
        response = await client.get(route, params={"q": q})
        response.raise_for_status()
        latencies.append((time.perf_counter() - started) * 1000)

    async def probe() -> None:
        while not stop.is_set():
            started = time.perf_counter()
            await client.get("/health")
            health.append((time.perf_counter() - started) * 1000)
            await asyncio.sleep(0.005)

    # прогрів: перші запити читають тексти з диска в кеш, його ми міряти не хочемо
    await asyncio.gather(*(one(q) for q in QUERIES[:CONCURRENCY]))
    latencies.clear()

    probing = asyncio.create_task(probe())
    for _ in range(ROUNDS):
        started = time.perf_counter()
        await asyncio.gather(*(one(q) for q in QUERIES[:CONCURRENCY]))
        walls.append(time.perf_counter() - started)
    stop.set()
    await probing
    return {
        "round_wall_s": statistics.median(walls),
        "rps": CONCURRENCY / statistics.median(walls),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "max_ms": max(latencies),
        "health_p50_ms": percentile(health, 50),
        "health_max_ms": max(health),
        "health_probes": float(len(health)),
    }


async def run_client(port: int) -> dict[str, dict[str, float]]:
    limits = httpx.Limits(max_connections=CONCURRENCY + 2)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=60, limits=limits
    ) as client:
        for _ in range(100):  # чекаємо, поки сервер завантажить індекс
            try:
                if (await client.get("/health")).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.2)
        return {
            route: await measure(client, route)
            for route in ("/search_def", "/search_async", "/search_thread")
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--serve", action="store_true", help="службовий режим: лише сервер")
    args = parser.parse_args()

    if args.serve:
        serve(args.index, args.corpus, args.port)
        return

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    server = subprocess.Popen(
        [
            sys.executable,
            __file__,
            str(args.index),
            str(args.corpus),
            "--serve",
            "--port",
            str(args.port),
        ],
        env=env,
    )
    try:
        results = asyncio.run(run_client(args.port))
    finally:
        server.kill()
        server.wait()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        "| маршрут | запитів/с | p50, мс | p95, мс | max, мс | /health p50, мс | /health max, мс |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for route, r in results.items():
        print(
            f"| {route} | {r['rps']:.1f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} "
            f"| {r['max_ms']:.0f} | {r['health_p50_ms']:.1f} | {r['health_max_ms']:.0f} |"
        )


if __name__ == "__main__":
    main()
