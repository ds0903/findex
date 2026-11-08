"""Одне завантаження сторінки: таймаут, повтори з backoff і jitter, Retry-After."""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from findex.crawl.limiter import Limiter
from findex.crawl.urls import host_of

log = logging.getLogger(__name__)

TIMEOUT = "timeout"


class OffSiteError(Exception):
    """Редірект повів на хост, якого не дозволено обходити."""


@dataclass(frozen=True, slots=True)
class FetchConfig:
    timeout: float = 10.0  # на один запит (з'єднання, заголовки й тіло разом)
    max_retries: int = 3  # повторів після першої спроби
    backoff_base: float = 0.5  # перша пауза до jitter, потім x2, x4, ...
    backoff_cap: float = 20.0
    max_retry_after: float = 30.0  # довше за це сервер просити не може
    max_bytes: int = 2_000_000  # більші сторінки не читаємо


DEFAULT_FETCH = FetchConfig()


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str  # після редіректів
    status: int | None  # None: відповіді не було (таймаут, обрив)
    body: str | None
    content_type: str
    nbytes: int
    seconds: float  # разом з повторами й паузами між ними
    attempts: int
    error: str | None = None
    retry_after: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def is_html(self) -> bool:
        return self.content_type.split(";")[0].strip().lower() in ("text/html", "text/plain")


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After це або число секунд, або дата HTTP. Повертає секунди чи None."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def should_retry(result: FetchResult) -> bool:
    """Лише 429, 5xx і таймаут. Інші 4xx (404, 403, 410...) повторювати марно."""
    if result.error == TIMEOUT:
        return True
    return result.status == 429 or (result.status is not None and 500 <= result.status < 600)


def backoff_delay(
    retry_number: int,
    config: FetchConfig,
    retry_after: float | None = None,
    rng: Callable[[], float] = random.random,
) -> float:
    """Скільки чекати перед повтором номер retry_number (1, 2, 3).

    Експоненційна межа base * 2^(n-1) з обрізкою cap, і з неї випадкова частка (jitter),
    щоб воркери, яких одночасно відшила одна помилка, не повернулись усі в ту саму мить.
    Якщо сервер сказав Retry-After, чекаємо не менше цього (але не довше max_retry_after).
    """
    ceiling = min(config.backoff_cap, config.backoff_base * 2 ** (retry_number - 1))
    delay = rng() * ceiling
    if retry_after is not None:
        delay = max(delay, min(retry_after, config.max_retry_after))
    return delay


async def _attempt(
    client: httpx.AsyncClient, url: str, config: FetchConfig, limiter: Limiter | None
) -> FetchResult:
    """Одна спроба. Очікувані мережеві помилки стають результатом, а не винятком."""
    started = time.perf_counter()
    status: int | None = None
    body: str | None = None
    ctype = ""
    nbytes = 0
    final_url = url
    retry_after: float | None = None
    error: str | None = None

    slot = limiter.slot(host_of(url)) if limiter else nullcontext()
    try:
        async with slot:
            async with asyncio.timeout(config.timeout):
                async with client.stream("GET", url) as response:
                    status = response.status_code
                    final_url = str(response.url)
                    ctype = response.headers.get("content-type", "")
                    retry_after = parse_retry_after(response.headers.get("retry-after"))
                    if status == 200 and ctype.lower().startswith("text/"):
                        chunks: list[bytes] = []
                        async for chunk in response.aiter_bytes():
                            nbytes += len(chunk)
                            if nbytes > config.max_bytes:
                                error = f"сторінка більша за {config.max_bytes} байт"
                                break
                            chunks.append(chunk)
                        if error is None:
                            body = b"".join(chunks).decode(
                                response.encoding or "utf-8", errors="replace"
                            )
    except (TimeoutError, httpx.TimeoutException):
        error = TIMEOUT
    except httpx.TooManyRedirects:
        error = "забагато редіректів"
    except OffSiteError as exc:
        error = str(exc)
    except httpx.HTTPError as exc:  # обрив, DNS, відмова у з'єднанні
        error = f"{type(exc).__name__}: {exc}"
    # CancelledError сюди не потрапляє: це BaseException, і ловити його не можна

    return FetchResult(
        url=url,
        final_url=final_url,
        status=status,
        body=body,
        content_type=ctype,
        nbytes=nbytes,
        seconds=time.perf_counter() - started,
        attempts=1,
        error=error,
        retry_after=retry_after,
    )


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    config: FetchConfig = DEFAULT_FETCH,
    limiter: Limiter | None = None,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> FetchResult:
    """Завантажує url. Повертає результат із статусом, байтами й часом (ніколи не кидає
    мережевих помилок). Пауза між повторами не тримає слот у Limiter."""
    started = time.perf_counter()
    attempt = 1
    while True:
        result = await _attempt(client, url, config, limiter)
        if not should_retry(result) or attempt > config.max_retries:
            break
        delay = backoff_delay(attempt, config, result.retry_after, rng)
        log.warning(
            "повтор %d/%d для %s: %s, чекаю %.2f с",
            attempt,
            config.max_retries,
            url,
            result.error or result.status,
            delay,
        )
        await sleep(delay)
        attempt += 1
    return replace(result, attempts=attempt, seconds=time.perf_counter() - started)
