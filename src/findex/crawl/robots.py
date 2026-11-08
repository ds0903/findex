import asyncio
import logging
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawl.fetch import DEFAULT_FETCH, FetchConfig, fetch
from findex.crawl.limiter import Limiter
from findex.crawl.urls import host_of, origin_of

log = logging.getLogger(__name__)

_DENY_ALL = ["User-agent: *", "Disallow: /"]


class RobotsCache:
    """robots.txt окремо для кожного хоста; відповідь завантажується один раз.

    Правила, якщо файл недоступний:
    - 200: беремо правила з файлу;
    - 401/403: усе заборонено (так само робить robotparser з stdlib);
    - інші 4xx (зазвичай 404): файлу нема, отже обмежень нема;
    - 5xx чи мережева помилка після повторів: нічого не знаємо, тому краще нічого не качати.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        config: FetchConfig = DEFAULT_FETCH,
        limiter: Limiter | None = None,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._config = config
        self._limiter = limiter
        self._rules: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        rules = await self._rules_for(url)
        return rules.can_fetch(self._user_agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        rules = await self._rules_for(url)
        delay = rules.crawl_delay(self._user_agent)
        return None if delay is None else float(delay)

    async def _rules_for(self, url: str) -> RobotFileParser:
        origin = origin_of(url)
        # замок на хост: десять воркерів, що одночасно дійшли до нового хоста,
        # завантажують robots.txt один раз, а не десять
        async with self._locks.setdefault(origin, asyncio.Lock()):
            if origin not in self._rules:
                self._rules[origin] = await self._load(origin)
                delay = self._rules[origin].crawl_delay(self._user_agent)
                if delay is not None and self._limiter is not None:
                    self._limiter.set_host_delay(host_of(origin), float(delay))
        return self._rules[origin]

    async def _load(self, origin: str) -> RobotFileParser:
        result = await fetch(self._client, f"{origin}/robots.txt", self._config, self._limiter)
        rules = RobotFileParser()
        if result.status == 200 and result.body is not None:
            rules.parse(result.body.splitlines())
            log.info("robots.txt %s: завантажено, %d байт", origin, result.nbytes)
        elif result.status in (401, 403):
            rules.parse(_DENY_ALL)
            log.warning("robots.txt %s: %s, обхід заборонено", origin, result.status)
        elif result.status is not None and 400 <= result.status < 500:
            rules.parse([])  # порожній файл: обмежень нема
            log.info("robots.txt %s: %s, обмежень нема", origin, result.status)
        else:
            rules.parse(_DENY_ALL)
            log.warning(
                "robots.txt %s недоступний (%s), обхід пропускаю",
                origin,
                result.error or result.status,
            )
        rules.modified()  # без цього can_fetch вважає, що файл ще не читали, і забороняє все
        return rules
