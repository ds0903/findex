import asyncio
from collections import Counter
from contextlib import aclosing

import httpx
import pytest

from findex.crawl import CrawlConfig, CrawlStats, FetchConfig, Page, crawl

HTML = {"content-type": "text/html; charset=utf-8"}
FAST = CrawlConfig(min_delay=0.0, fetch=FetchConfig(backoff_base=0.0, timeout=1.0))


def html(title: str, *links: str, head: str = "") -> str:
    anchors = "".join(f'<a href="{link}">l</a>' for link in links)
    body = f"<p>about {title}</p>{anchors}"
    return f"<html><head><title>{title}</title>{head}</head><body>{body}</body></html>"


class Site:
    """Фальшивий сайт для MockTransport: path -> html чи готова відповідь. Веде облік запитів."""

    def __init__(self, pages: dict[str, str | httpx.Response], robots: str | None = None):
        self.pages = pages
        self.robots = robots
        self.hits: Counter[str] = Counter()
        self.agents: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = request.url.path + (f"?{request.url.query.decode()}" if request.url.query else "")
        self.hits[f"{request.url.host}{key}"] += 1
        self.agents.add(request.headers.get("user-agent", ""))
        if request.url.path == "/robots.txt":
            if self.robots is None:
                return httpx.Response(404)
            return httpx.Response(200, text=self.robots, headers={"content-type": "text/plain"})
        page = self.pages.get(key)
        if page is None:
            return httpx.Response(404)
        if isinstance(page, httpx.Response):  # копія: одну відповідь можуть брати кілька разів
            return httpx.Response(page.status_code, headers=page.headers, content=page.content)
        return httpx.Response(200, text=page, headers=HTML)

    def fetched(self) -> list[str]:
        return sorted(k for k in self.hits if not k.endswith("/robots.txt"))


def run_crawl(
    site: Site, seeds: list[str], config: CrawlConfig = FAST
) -> tuple[list[Page], CrawlStats]:
    stats = CrawlStats()

    async def go() -> list[Page]:
        async with aclosing(
            crawl(seeds, config, stats=stats, transport=httpx.MockTransport(site))
        ) as pages:
            return [page async for page in pages]

    return asyncio.run(go()), stats


def urls(pages: list[Page]) -> set[str]:
    return {page.url for page in pages}


def test_follows_links_and_extracts_text():
    site = Site({"/": html("Home", "/a", "b"), "/a": html("A"), "/b": html("B")})
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert urls(pages) == {"http://s.test/", "http://s.test/a", "http://s.test/b"}
    home = next(p for p in pages if p.url == "http://s.test/")
    assert home.title == "Home" and "about Home" in home.text
    assert home.depth == 0 and home.status == 200 and home.nbytes > 0
    assert stats.pages == 3 and stats.errors == 0


def test_same_page_is_fetched_once_whatever_the_link_looks_like():
    site = Site(
        {
            "/": html("Home", "/a", "/a/", "/a#top", "http://S.test/a", "/s?b=2&a=1", "/s?a=1&b=2"),
            "/a": html("A", "/", "/s?a=1&b=2"),
            "/s?a=1&b=2": html("S", "/a"),
        }
    )
    pages, _ = run_crawl(site, ["http://s.test/", "http://s.test"])
    assert len(pages) == 3
    assert all(count == 1 for key, count in site.hits.items() if key != "s.test/robots.txt")


def test_stays_on_allowed_hosts():
    site = Site(
        {"/": html("Home", "http://evil.test/x", "/a", "mailto:a@b.c", "/f.pdf"), "/a": html("A")}
    )
    pages, _ = run_crawl(site, ["http://s.test/"])
    assert urls(pages) == {"http://s.test/", "http://s.test/a"}
    assert site.fetched() == ["s.test/", "s.test/a"]


def test_extra_allowed_host():
    site = Site({"/": html("Home", "http://docs.test/x"), "/x": html("X")})
    config = CrawlConfig(min_delay=0.0, allowed_hosts=frozenset({"docs.test"}))
    pages, _ = run_crawl(site, ["http://s.test/"], config)
    assert "http://docs.test/x" in urls(pages)


def test_respects_robots_txt():
    site = Site(
        {"/": html("Home", "/ok", "/private/x"), "/ok": html("OK"), "/private/x": html("X")},
        robots="User-agent: *\nDisallow: /private\n",
    )
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert urls(pages) == {"http://s.test/", "http://s.test/ok"}
    assert "s.test/private/x" not in site.hits
    assert stats.robots_blocked == 1


def test_robots_can_forbid_everything():
    site = Site({"/": html("Home")}, robots="User-agent: *\nDisallow: /\n")
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert pages == [] and stats.robots_blocked == 1


def test_honest_user_agent_is_sent():
    site = Site({"/": html("Home")})
    run_crawl(site, ["http://s.test/"])
    (agent,) = site.agents
    assert agent.startswith("findex-crawler/") and "https://github.com/ds0903/findex" in agent


def test_max_pages_is_a_hard_ceiling():
    site = Site({f"/{i}": html(str(i), f"/{i + 1}", f"/{i + 2}") for i in range(100)})
    site.pages["/"] = html("Home", "/0", "/1")
    pages, stats = run_crawl(site, ["http://s.test/"], CrawlConfig(max_pages=7, min_delay=0.0))
    assert len(pages) == 7 and stats.pages == 7
    assert len(site.fetched()) == 7  # зайвих запитів на сайт нема


def test_max_depth():
    site = Site(
        {"/": html("Home", "/1"), "/1": html("One", "/2"), "/2": html("Two", "/3"), "/3": html("T")}
    )
    pages, _ = run_crawl(site, ["http://s.test/"], CrawlConfig(max_depth=2, min_delay=0.0))
    assert [p.depth for p in sorted(pages, key=lambda p: p.depth)] == [0, 1, 2]


def test_errors_are_counted_and_do_not_stop_the_crawl():
    site = Site(
        {
            "/": html("Home", "/gone", "/boom", "/fine"),
            "/boom": httpx.Response(500),
            "/fine": html("Fine"),
        }
    )
    config = CrawlConfig(min_delay=0.0, fetch=FetchConfig(max_retries=1, backoff_base=0.0))
    pages, stats = run_crawl(site, ["http://s.test/"], config)
    assert urls(pages) == {"http://s.test/", "http://s.test/fine"}
    assert stats.errors == 2  # 404 і 500
    assert stats.retries == 1  # повторили лише 500
    assert site.hits["s.test/boom"] == 2 and site.hits["s.test/gone"] == 1


def test_429_with_retry_after_is_recovered():
    answers = iter([httpx.Response(429, headers={"Retry-After": "0"}), None])

    class Flaky(Site):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/slowdown" and (first := next(answers)) is not None:
                return first
            return super().__call__(request)

    site = Flaky({"/": html("Home", "/slowdown"), "/slowdown": html("Slow")})
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert "http://s.test/slowdown" in urls(pages)
    assert stats.errors == 0 and stats.retries == 1


def test_redirect_to_other_host_is_refused():
    site = Site(
        {
            "/": html("Home", "/jump", "/ok"),
            "/jump": httpx.Response(302, headers={"Location": "http://evil.test/x"}),
            "/ok": html("OK"),
        }
    )
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert urls(pages) == {"http://s.test/", "http://s.test/ok"}
    assert "evil.test/x" not in site.hits
    assert stats.errors == 1


def test_redirect_loop_is_one_error():
    site = Site(
        {"/": html("Home", "/loop"), "/loop": httpx.Response(302, headers={"Location": "/loop"})}
    )
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert len(pages) == 1 and stats.errors == 1


def test_noindex_page_is_not_returned_but_links_are_followed():
    head = '<meta name="robots" content="noindex">'
    site = Site(
        {
            "/": html("Home", "/hidden"),
            "/hidden": html("Hidden", "/deep", head=head),
            "/deep": html("Deep"),
        }
    )
    pages, _ = run_crawl(site, ["http://s.test/"])
    assert urls(pages) == {"http://s.test/", "http://s.test/deep"}


def test_non_html_and_bad_seed():
    site = Site(
        {
            "/": html("Home", "/data"),
            "/data": httpx.Response(
                200, content=b"{}", headers={"content-type": "application/json"}
            ),
        }
    )
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert len(pages) == 1 and stats.errors == 0
    with pytest.raises(ValueError):
        run_crawl(site, ["ftp://s.test/"])
    with pytest.raises(ValueError):
        CrawlConfig(max_pages=0)


def test_requests_run_concurrently_but_within_limits():
    now = peak = 0

    class Slow(Site):
        async def __call__(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
            nonlocal now, peak
            now += 1
            peak = max(peak, now)
            await asyncio.sleep(0.05)
            now -= 1
            return super().__call__(request)

    site = Slow(
        {
            "/": html("Home", *[f"/p{i}" for i in range(39)]),
            **{f"/p{i}": html(str(i)) for i in range(39)},
        }
    )
    config = CrawlConfig(max_pages=40, concurrency=10, per_host=4, min_delay=0.0)
    pages, _ = run_crawl(site, ["http://s.test/"], config)
    assert len(pages) == 40
    assert peak == 4  # ліміт на хост, хоч загальний дозволяє 10


def test_concurrency_speeds_up_waiting():
    class Slow(Site):
        async def __call__(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
            await asyncio.sleep(0.05)
            return super().__call__(request)

    pages = {f"/p{i}": html(str(i)) for i in range(30)}
    pages["/"] = html("Home", *pages)
    results = {}
    for concurrency in (1, 10):
        config = CrawlConfig(
            max_pages=31, concurrency=concurrency, per_host=concurrency, min_delay=0.0
        )
        _, stats = run_crawl(Slow(dict(pages)), ["http://s.test/"], config)
        results[concurrency] = stats.elapsed
    assert results[10] < results[1] / 3  # 31 * 0.05 = 1.5 с послідовно, ~0.2 с при 10


def test_early_exit_cancels_workers_and_closes_client():
    site = Site({f"/{i}": html(str(i), f"/{i + 1}") for i in range(50)})
    site.pages["/"] = html("Home", "/0")

    async def go() -> None:
        async with aclosing(
            crawl(
                ["http://s.test/"],
                CrawlConfig(max_pages=50, min_delay=0.0),
                transport=httpx.MockTransport(site),
            )
        ) as pages:
            async for page in pages:
                if page.depth == 2:
                    break
        # після виходу нема жодної живої задачі, крім цієї
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(go())
    assert len(site.fetched()) < 50


def test_bug_in_one_page_does_not_kill_the_crawl(monkeypatch: pytest.MonkeyPatch):
    import findex.crawl.crawler as module

    real = module.parse_html

    def flaky_parse(text: str):
        if "about Bad" in text:
            raise RuntimeError("розбір зламався")
        return real(text)

    monkeypatch.setattr(module, "parse_html", flaky_parse)
    site = Site({"/": html("Home", "/bad", "/good"), "/bad": html("Bad"), "/good": html("Good")})
    pages, stats = run_crawl(site, ["http://s.test/"])
    assert urls(pages) == {"http://s.test/", "http://s.test/good"}
    assert stats.errors == 1
