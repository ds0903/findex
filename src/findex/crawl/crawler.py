"""Асинхронний обхід сайту: черга адрес, воркери в TaskGroup, ліміти, robots.txt."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import httpx

from findex.crawl.counters import CrawlStats
from findex.crawl.fetch import FetchConfig, OffSiteError, fetch
from findex.crawl.limiter import Limiter
from findex.crawl.pages import parse_html
from findex.crawl.robots import RobotsCache
from findex.crawl.urls import canonicalize, host_of, looks_like_page

log = logging.getLogger(__name__)
# статус кожного URL окремим логером: CLI відправляє його у crawl.log
url_log = logging.getLogger("findex.crawl.urls")

BIG_PAGE = 200_000  # символів; більші сторінки розбираємо в потоці, щоб не тримати цикл подій


def default_user_agent() -> str:
    try:
        number = version("findex")
    except PackageNotFoundError:
        number = "dev"
    return f"findex-crawler/{number} (+https://github.com/ds0903/findex/issues)"


@dataclass(frozen=True, slots=True)
class CrawlConfig:
    max_pages: int = 100  # стеля на кількість завантажених адрес
    concurrency: int = 5  # скільки запитів одночасно (загальний семафор)
    per_host: int = 2  # скільки з них до одного хоста
    min_delay: float = 0.5  # пауза між початками запитів до одного хоста, секунди
    max_depth: int = 10
    allowed_hosts: frozenset[str] = frozenset()  # порожньо: лише хости початкових адрес
    respect_robots: bool = True
    user_agent: str = field(default_factory=default_user_agent)
    fetch: FetchConfig = FetchConfig()

    def __post_init__(self) -> None:
        if self.max_pages < 1 or self.concurrency < 1 or self.per_host < 1:
            raise ValueError("max_pages, concurrency і per_host мають бути не менші за 1")


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    depth: int
    status: int
    nbytes: int
    seconds: float
    fetched_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


DEFAULT_CRAWL = CrawlConfig()

_DONE = object()


class _Frontier:
    """Черга адрес зі множиною побачених і лічильниками, за якими видно, що обхід закінчився."""

    def __init__(self, stats: CrawlStats) -> None:
        self.queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self.seen: set[str] = set()
        self.pending = 0  # у черзі + у роботі
        self.claimed = 0  # адрес, для яких запит уже дозволено (ліміт max_pages)
        self.completed = 0  # з них завершених
        self._stats = stats

    def add(self, url: str, depth: int) -> bool:
        if url in self.seen:
            return False
        self.seen.add(url)
        self.pending += 1
        self.queue.put_nowait((url, depth))
        self._stats.queued = self.queue.qsize()
        return True


async def crawl(
    seeds: list[str],
    config: CrawlConfig = DEFAULT_CRAWL,
    *,
    stats: CrawlStats | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncGenerator[Page]:
    """Обходить сайт від початкових адрес і по одній віддає завантажені сторінки.

    Це асинхронний генератор: сторінки не накопичуються, споживач бере їх у міру надходження.
    Закривайте його через contextlib.aclosing: тоді при раннім виході воркери скасуються,
    а клієнт закриється. Помилки окремих адрес (404, таймаут) не виняток: вони йдуть
    у stats.errors і в лог.
    """
    stats = stats if stats is not None else CrawlStats()
    start_urls = [canonicalize(seed) for seed in seeds]
    if not start_urls or None in start_urls:
        raise ValueError(f"початкові адреси мають бути http(s): {seeds}")
    roots = [u for u in start_urls if u is not None]
    allowed = config.allowed_hosts | {host_of(u) for u in roots}

    async def guard_host(request: httpx.Request) -> None:
        # спрацьовує і на кожному редіректі: за межі дозволених хостів не виходимо
        if request.url.netloc.decode("ascii").lower() not in allowed:
            raise OffSiteError(f"редірект на чужий хост {request.url.host}")

    # httpx будує SSL-контекст синхронно (читає ~150 CA-сертифікатів, ≈0.2 с на Windows).
    # Прямо в корутині це зупинило б цикл подій: asyncio.run(debug=True) це показує.
    # Тому контекст робимо в потоці, а клієнту віддаємо готовий.
    verify = await asyncio.to_thread(httpx.create_ssl_context) if transport is None else True

    frontier = _Frontier(stats)
    out: asyncio.Queue[object] = asyncio.Queue(maxsize=config.concurrency * 2)
    finished = asyncio.Event()
    limiter = Limiter(config.concurrency, config.per_host, config.min_delay, stats)

    async with httpx.AsyncClient(
        headers={"User-Agent": config.user_agent},
        timeout=config.fetch.timeout,
        verify=verify,
        follow_redirects=True,
        max_redirects=5,
        limits=httpx.Limits(max_connections=config.concurrency),
        transport=transport,
        event_hooks={"request": [guard_host]},
    ) as client:
        robots = RobotsCache(client, config.user_agent, config.fetch, limiter)

        def check_finished() -> None:
            if frontier.pending == 0 or frontier.completed >= config.max_pages:
                finished.set()

        async def handle(url: str, depth: int) -> None:
            if frontier.claimed >= config.max_pages:
                return
            if config.respect_robots and not await robots.allowed(url):
                stats.robots_blocked += 1
                url_log.info("robots   %s", url)
                return
            if frontier.claimed >= config.max_pages:  # пока чекали robots.txt, ліміт вичерпали
                return
            frontier.claimed += 1
            try:
                result = await fetch(client, url, config.fetch, limiter)
                stats.retries += result.attempts - 1
                stats.nbytes += result.nbytes
                if result.error is not None or result.status != 200:
                    stats.errors += 1
                    url_log.warning(
                        "ПОМИЛКА  %s status=%s error=%s tries=%d %.2fs",
                        url,
                        result.status,
                        result.error,
                        result.attempts,
                        result.seconds,
                    )
                    return
                url_log.info(
                    "%s  %s %d B %.2fs tries=%d",
                    result.status,
                    url,
                    result.nbytes,
                    result.seconds,
                    result.attempts,
                )
                if not result.is_html or result.body is None:
                    return
                if len(result.body) > BIG_PAGE:
                    parsed = await asyncio.to_thread(parse_html, result.body)
                else:
                    parsed = parse_html(result.body)
                if not parsed.nofollow and depth < config.max_depth:
                    for link in parsed.links:
                        target = canonicalize(link, base=result.final_url)
                        if target and host_of(target) in allowed and looks_like_page(target):
                            frontier.add(target, depth + 1)
                if not parsed.noindex:
                    page = Page(
                        url=url,
                        title=parsed.title,
                        text=parsed.text,
                        depth=depth,
                        status=result.status,
                        nbytes=result.nbytes,
                        seconds=round(result.seconds, 4),
                        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    )
                    stats.pages += 1
                    await out.put(page)
            finally:
                frontier.completed += 1

        async def worker() -> None:
            while True:
                url, depth = await frontier.queue.get()
                stats.queued = frontier.queue.qsize()
                try:
                    await handle(url, depth)
                except Exception:  # одна погана сторінка не має зупиняти обхід
                    stats.errors += 1
                    log.exception("збій при обробці %s", url)
                finally:
                    frontier.pending -= 1
                    check_finished()

        async def closer() -> None:
            await finished.wait()
            await out.put(_DONE)

        for root in roots:
            frontier.add(root, 0)
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(worker()) for _ in range(config.concurrency * 2)]
            tasks.append(group.create_task(closer()))
            try:
                while (item := await out.get()) is not _DONE:
                    assert isinstance(item, Page)
                    yield item
            except GeneratorExit:
                # споживач закрив генератор (break). Без цього except TaskGroup загорнув би
                # GeneratorExit у BaseExceptionGroup, і aclose() впав би з цією групою.
                pass
            finally:
                # воркери крутяться вічно (чекають на чергу), тож TaskGroup сам не завершиться
                for task in tasks:
                    task.cancel()
