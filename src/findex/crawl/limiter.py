import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from findex.crawl.counters import CrawlStats


class Limiter:
    """Скільки запитів може летіти одночасно.

    Три обмеження: один семафор на всі запити, окремий семафор на хост і мінімальна пауза
    між початками запитів до того самого хоста. Порядок такий: спершу місце на хості
    і пауза, потім загальний слот. Інакше воркер, що чекає на зайнятий хост, тримав би
    загальний слот і гальмував інші хости.
    """

    def __init__(
        self,
        total: int,
        per_host: int,
        min_delay: float = 0.0,
        stats: CrawlStats | None = None,
    ) -> None:
        self._total = asyncio.Semaphore(total)
        self._per_host = per_host
        self._min_delay = min_delay
        self._hosts: dict[str, asyncio.Semaphore] = {}
        self._next_start: dict[str, float] = {}
        self._host_delay: dict[str, float] = {}  # Crawl-delay з robots.txt
        self._stats = stats or CrawlStats()

    def set_host_delay(self, host: str, delay: float) -> None:
        self._host_delay[host] = delay

    @asynccontextmanager
    async def slot(self, host: str) -> AsyncGenerator[None]:
        per_host = self._hosts.setdefault(host, asyncio.Semaphore(self._per_host))
        async with per_host:
            await self._wait_turn(host)
            async with self._total:
                self._stats.in_flight += 1
                try:
                    yield
                finally:
                    self._stats.in_flight -= 1

    async def _wait_turn(self, host: str) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        start = max(now, self._next_start.get(host, 0.0))
        # між читанням і записом немає await, тож два воркери не отримають той самий момент
        self._next_start[host] = start + max(self._min_delay, self._host_delay.get(host, 0.0))
        if start > now:
            await asyncio.sleep(start - now)
