import asyncio
from collections.abc import Callable

import httpx

from findex.crawl import CrawlStats, FetchConfig, Limiter, RobotsCache

UA = "findex-test/1.0 (+https://example.test/contact)"
FAST = FetchConfig(max_retries=0)  # без повторів, щоб тест не чекав

ROBOTS = "User-agent: *\nDisallow: /private\nCrawl-delay: 2\n"


def robots_cache(handler: Callable[[httpx.Request], httpx.Response], limiter=None):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, RobotsCache(client, UA, FAST, limiter)


# --- Limiter ---


def test_per_host_limit():
    now = peak = 0

    async def visit(limiter: Limiter) -> None:
        nonlocal now, peak
        async with limiter.slot("a.test"):
            now += 1
            peak = max(peak, now)
            await asyncio.sleep(0.01)
            now -= 1

    async def go() -> None:
        limiter = Limiter(total=10, per_host=2)
        await asyncio.gather(*(visit(limiter) for _ in range(12)))

    asyncio.run(go())
    assert peak == 2


def test_global_limit_across_hosts():
    now = peak = 0

    async def visit(limiter: Limiter, host: str) -> None:
        nonlocal now, peak
        async with limiter.slot(host):
            now += 1
            peak = max(peak, now)
            await asyncio.sleep(0.01)
            now -= 1

    async def go() -> None:
        limiter = Limiter(total=3, per_host=3)
        await asyncio.gather(*(visit(limiter, f"h{i % 6}.test") for i in range(24)))

    asyncio.run(go())
    assert peak == 3


def test_min_delay_between_requests_to_same_host():
    starts: list[float] = []

    async def visit(limiter: Limiter) -> None:
        async with limiter.slot("a.test"):
            starts.append(asyncio.get_running_loop().time())

    async def go() -> None:
        limiter = Limiter(total=10, per_host=10, min_delay=0.1)
        await asyncio.gather(*(visit(limiter) for _ in range(4)))

    asyncio.run(go())
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    # asyncio на Windows може розбудити таймер на ~15 мс раніше (роздільність годинника)
    assert len(gaps) == 3 and all(gap >= 0.08 for gap in gaps)


def test_other_hosts_are_not_slowed_by_delay():
    starts: dict[str, float] = {}

    async def visit(limiter: Limiter, host: str) -> None:
        async with limiter.slot(host):
            starts[host] = asyncio.get_running_loop().time()

    async def go() -> None:
        limiter = Limiter(total=10, per_host=1, min_delay=0.2)
        t0 = asyncio.get_running_loop().time()
        await asyncio.gather(*(visit(limiter, h) for h in ("a.test", "b.test", "c.test")))
        assert max(starts.values()) - t0 < 0.1

    asyncio.run(go())


def test_in_flight_counter():
    seen: list[int] = []

    async def go() -> None:
        stats = CrawlStats()
        limiter = Limiter(total=5, per_host=5, stats=stats)

        async def visit() -> None:
            async with limiter.slot("a.test"):
                await asyncio.sleep(0.01)
                seen.append(stats.in_flight)

        await asyncio.gather(*(visit() for _ in range(3)))
        assert stats.in_flight == 0

    asyncio.run(go())
    assert max(seen) == 3


# --- RobotsCache ---


def test_robots_rules_apply():
    client, robots = robots_cache(lambda r: httpx.Response(200, text=ROBOTS))

    async def go() -> None:
        async with client:
            assert await robots.allowed("http://s.test/page")
            assert not await robots.allowed("http://s.test/private/secret")
            assert await robots.crawl_delay("http://s.test/page") == 2.0

    asyncio.run(go())


def test_robots_is_fetched_once_per_host_even_when_asked_concurrently():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text=ROBOTS)

    client, robots = robots_cache(handler)

    async def go() -> None:
        async with client:
            await asyncio.gather(*(robots.allowed(f"http://s.test/p{i}") for i in range(15)))
            await robots.allowed("http://other.test/x")
            await robots.allowed("http://other.test/y")

    asyncio.run(go())
    assert sorted(calls) == ["http://other.test/robots.txt", "http://s.test/robots.txt"]


def test_ports_are_separate_hosts():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.netloc.decode())
        return httpx.Response(404)

    client, robots = robots_cache(handler)

    async def go() -> None:
        async with client:
            await robots.allowed("http://s.test:8000/a")
            await robots.allowed("http://s.test:9000/a")

    asyncio.run(go())
    assert sorted(calls) == ["s.test:8000", "s.test:9000"]


def allowed_when(status: int) -> bool:
    client, robots = robots_cache(lambda r: httpx.Response(status, text="User-agent: *\nAllow: /"))

    async def go() -> bool:
        async with client:
            return await robots.allowed("http://s.test/page")

    return asyncio.run(go())


def test_missing_robots_means_allowed():
    assert allowed_when(404)
    assert allowed_when(410)


def test_forbidden_or_broken_robots_means_not_allowed():
    assert not allowed_when(401)
    assert not allowed_when(403)
    assert not allowed_when(500)


def test_network_failure_on_robots_means_not_allowed():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, robots = robots_cache(handler)

    async def go() -> bool:
        async with client:
            return await robots.allowed("http://s.test/page")

    assert asyncio.run(go()) is False


def test_crawl_delay_is_passed_to_limiter():
    starts: list[float] = []

    async def go() -> None:
        limiter = Limiter(total=5, per_host=5)
        client, robots = robots_cache(
            lambda r: httpx.Response(200, text="User-agent: *\nCrawl-delay: 1"), limiter
        )  # robotparser розуміє лише цілі секунди
        async with client:
            await robots.allowed("http://s.test/a")  # завантажує robots.txt, ставить паузу
            for _ in range(2):
                async with limiter.slot("s.test"):
                    starts.append(asyncio.get_running_loop().time())

    asyncio.run(go())
    assert starts[1] - starts[0] >= 0.95
