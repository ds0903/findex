import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.crawl import FetchConfig, canonicalize, fetch
from findex.crawl.fetch import DEFAULT_FETCH, backoff_delay, parse_retry_after, should_retry
from findex.crawl.urls import host_of, looks_like_page, origin_of

HTML = {"content-type": "text/html; charset=utf-8"}


def make_client(handler: Callable[[httpx.Request], Any], **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


class Sleeps:
    """Підміна asyncio.sleep: запам'ятовує паузи й не чекає по-справжньому."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def run_fetch(handler, url="http://site.test/a", config=DEFAULT_FETCH, rng=lambda: 1.0, **client):
    sleeps = Sleeps()

    async def go():
        async with make_client(handler, **client) as c:
            return await fetch(c, url, config, sleep=sleeps, rng=rng)

    return asyncio.run(go()), sleeps


# --- canonicalize ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM/a", "http://example.com/a"),
        ("http://example.com", "http://example.com/"),
        ("http://example.com/a/", "http://example.com/a"),
        ("http://example.com/a#section", "http://example.com/a"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:8000/a", "http://example.com:8000/a"),
        ("http://example.com/s?b=2&a=1", "http://example.com/s?a=1&b=2"),
        ("http://user:pw@example.com/a", "http://example.com/a"),
        ("http://[::1]:8000/a", "http://[::1]:8000/a"),
    ],
)
def test_canonicalize(raw: str, expected: str):
    assert canonicalize(raw) == expected


@pytest.mark.parametrize("raw", ["mailto:a@b.c", "javascript:void(0)", "ftp://x.test/a", "/a", ""])
def test_canonicalize_rejects_non_http(raw: str):
    assert canonicalize(raw) is None


def test_canonicalize_relative_to_base():
    assert canonicalize("../b?y=1#x", base="http://s.test/dir/a/") == "http://s.test/dir/b?y=1"
    assert canonicalize("//cdn.test/x", base="https://s.test/a") == "https://cdn.test/x"


@given(st.text(max_size=60))
def test_canonicalize_idempotent(path: str):
    once = canonicalize("http://Example.com/" + path)
    if once is not None:
        assert canonicalize(once) == once


def test_url_helpers():
    assert host_of("http://s.test:8000/a") == "s.test:8000"
    assert origin_of("https://s.test/a/b?x=1") == "https://s.test"
    assert looks_like_page("http://s.test/a.html")
    assert not looks_like_page("http://s.test/photo.JPG")


# --- retry-after, backoff ---


def test_parse_retry_after():
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # дата в минулому


def test_backoff_grows_and_is_capped():
    cfg = FetchConfig(backoff_base=1.0, backoff_cap=5.0)
    assert [backoff_delay(n, cfg, rng=lambda: 1.0) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 5.0]


def test_backoff_has_jitter():
    cfg = FetchConfig(backoff_base=1.0)
    assert backoff_delay(3, cfg, rng=lambda: 0.0) == 0.0
    assert backoff_delay(3, cfg, rng=lambda: 0.5) == 2.0
    values = {backoff_delay(3, cfg) for _ in range(20)}
    assert len(values) > 1 and all(0 <= v <= 4.0 for v in values)


def test_retry_after_raises_the_floor_but_is_capped():
    cfg = FetchConfig(backoff_base=0.1, max_retry_after=30)
    assert backoff_delay(1, cfg, retry_after=7, rng=lambda: 1.0) == 7
    assert backoff_delay(1, cfg, retry_after=9999, rng=lambda: 1.0) == 30


# --- fetch ---


def test_success_records_status_bytes_and_time():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="привіт".encode(), headers=HTML)

    result, sleeps = run_fetch(handler)
    assert result.ok and result.status == 200
    assert result.body == "привіт"
    assert result.nbytes == len("привіт".encode())
    assert result.seconds >= 0 and result.attempts == 1
    assert result.is_html
    assert sleeps.delays == []


def test_404_is_not_retried():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    result, sleeps = run_fetch(handler)
    assert result.status == 404 and not result.ok
    assert len(calls) == 1 and sleeps.delays == []


@pytest.mark.parametrize("status", [400, 401, 403, 410])
def test_other_4xx_are_not_retried(status: int):
    assert not should_retry(run_fetch(lambda r: httpx.Response(status))[0])


def test_503_is_retried_with_exponential_backoff():
    answers = iter([503, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(answers), content=b"ok", headers=HTML)

    result, sleeps = run_fetch(handler, config=FetchConfig(backoff_base=0.5))
    assert result.ok and result.attempts == 3
    assert sleeps.delays == [0.5, 1.0]  # rng=1.0: без jitter видно саму прогресію


def test_gives_up_after_three_retries():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    result, sleeps = run_fetch(handler)
    assert len(calls) == 4  # перша спроба + 3 повтори
    assert result.status == 500 and result.attempts == 4 and len(sleeps.delays) == 3


def test_429_waits_at_least_retry_after():
    answers = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])
    result, sleeps = run_fetch(lambda r: next(answers), config=FetchConfig(backoff_base=0.1))
    assert result.status == 200
    assert sleeps.delays == [7.0]


def test_retry_after_is_capped():
    answers = iter([httpx.Response(429, headers={"Retry-After": "86400"}), httpx.Response(200)])
    _, sleeps = run_fetch(lambda r: next(answers), config=FetchConfig(max_retry_after=12))
    assert sleeps.delays == [12.0]


def test_timeout_is_retried_then_reported():
    calls: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        await asyncio.sleep(1)
        return httpx.Response(200)

    result, _ = run_fetch(handler, config=FetchConfig(timeout=0.05, max_retries=2))
    assert len(calls) == 3
    assert result.error == "timeout" and result.status is None and not result.ok


def test_connection_error_is_reported_without_retry():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    result, sleeps = run_fetch(handler)
    assert len(calls) == 1 and sleeps.delays == []
    assert result.status is None and "ConnectError" in (result.error or "")


def test_redirect_loop_is_reported_without_retry():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(302, headers={"Location": "/a"})

    result, sleeps = run_fetch(handler, follow_redirects=True, max_redirects=5)
    assert result.error == "забагато редіректів"
    assert len(calls) == 6 and result.attempts == 1 and sleeps.delays == []


def test_non_text_body_is_not_read():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    result, _ = run_fetch(handler)
    assert result.status == 200 and result.body is None and result.nbytes == 0
    assert not result.is_html


def test_oversized_page_is_dropped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000, headers=HTML)

    result, _ = run_fetch(handler, config=FetchConfig(max_bytes=1000))
    assert result.body is None and "більша" in (result.error or "")


def test_cancellation_is_not_swallowed():
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200)

    async def go() -> None:
        async with make_client(handler) as client:
            task = asyncio.create_task(fetch(client, "http://site.test/a"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(go())
