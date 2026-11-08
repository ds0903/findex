"""Навантажувальний тест (лаба 7): RPS і p50/p95/p99 для одного URL.

    python scripts/load_test.py "http://127.0.0.1:8000/search?q=whale&k=5" -c 50 -z 30

Закритий цикл: -c клієнтів, кожен шле наступний запит, щойно прийшла відповідь, і так -z секунд.
Це заміна `oha` (інструмент із завдання), яку я міг запустити без сторонніх бінарників; формат
звіту той самий: RPS, p50, p95, p99. Запити, що не встигли завершитись до кінця вікна, не
враховуються (при повільному сервері це обрізає хвіст розподілу, тому в README вказано, скільки
запитів завершилось).
"""

import argparse
import asyncio
import json
import time

import httpx


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, round(p / 100 * (len(sorted_values) - 1)))
    return sorted_values[index]


async def run(url: str, concurrency: int, seconds: float) -> dict[str, float]:
    latencies: list[float] = []
    errors = 0
    deadline = time.perf_counter() + seconds

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:

        async def worker() -> None:
            nonlocal errors
            while time.perf_counter() < deadline:
                started = time.perf_counter()
                try:
                    response = await client.get(url)
                    ok = response.status_code == 200
                except httpx.HTTPError:
                    ok = False
                finished = time.perf_counter()
                if finished > deadline:
                    break  # не врахований: відповідь прийшла вже після вікна
                if ok:
                    latencies.append((finished - started) * 1000)
                else:
                    errors += 1

        began = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - began

    latencies.sort()
    return {
        "requests": len(latencies),
        "errors": errors,
        "seconds": round(elapsed, 1),
        "rps": round(len(latencies) / elapsed, 2),
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("-c", "--concurrency", type=int, default=50)
    parser.add_argument("-z", "--seconds", type=float, default=30)
    args = parser.parse_args()
    result = asyncio.run(run(args.url, args.concurrency, args.seconds))
    result.update(url=args.url, concurrency=args.concurrency)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
