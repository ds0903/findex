import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner, Result

import findex.cli as cli
from findex.cli import app
from findex.documents import iter_documents

runner = CliRunner()

SITE = {
    "/": '<title>Home</title><p>library</p><a href="/whale">w</a><a href="/ship">s</a>'
    '<a href="/private/x">p</a>',
    "/whale": "<title>Whale Book</title><p>the white whale swims in the deep sea</p>",
    "/ship": "<title>Ship Book</title><p>the ship sails across the sea</p>",
    "/private/x": "<title>Secret</title><p>secret</p>",
}
ROBOTS = "User-agent: *\nDisallow: /private\n"


def site(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text=ROBOTS, headers={"content-type": "text/plain"})
    body = SITE.get(request.url.path)
    if body is None:
        return httpx.Response(404)
    return httpx.Response(200, text=body, headers={"content-type": "text/html"})


@pytest.fixture(autouse=True)
def fake_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_TRANSPORT", httpx.MockTransport(site))


def run(*args: str | Path) -> Result:
    return runner.invoke(app, [str(a) for a in args], env={"COLUMNS": "200"})


def crawl(tmp_path: Path, *extra: str | Path) -> Result:
    return run(
        "crawl",
        "http://s.test/",
        "--out",
        tmp_path / "pages.jsonl",
        "--log",
        tmp_path / "crawl.log",
        "--delay",
        "0",
        *extra,
    )


def test_crawl_writes_jsonl(tmp_path: Path):
    result = crawl(tmp_path)
    assert result.exit_code == 0, result.output
    lines = (tmp_path / "pages.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert {r["url"] for r in records} == {
        "http://s.test/",
        "http://s.test/whale",
        "http://s.test/ship",
    }
    whale = next(r for r in records if r["title"] == "Whale Book")
    assert "white whale" in whale["text"] and whale["status"] == 200 and whale["nbytes"] > 0
    assert "сторінок:     3" in result.stdout


def test_crawl_log_has_status_of_every_url(tmp_path: Path):
    crawl(tmp_path)
    log = (tmp_path / "crawl.log").read_text(encoding="utf-8")
    assert "200  http://s.test/whale" in log
    assert "robots   http://s.test/private/x" in log
    assert "robots.txt http://s.test" in log


def test_crawl_stats_json(tmp_path: Path):
    result = crawl(tmp_path, "--stats-json", tmp_path / "stats.json")
    assert result.exit_code == 0, result.output
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["pages"] == 3 and stats["errors"] == 0 and stats["robots_blocked"] == 1
    assert stats["peak_rss_mb"] > 0 and stats["slow_callbacks"] is None


def test_crawl_max_pages(tmp_path: Path):
    result = crawl(tmp_path, "--max-pages", "2")
    assert result.exit_code == 0, result.output
    assert len((tmp_path / "pages.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_crawl_debug_reports_blocked_loop(tmp_path: Path):
    result = crawl(tmp_path, "--debug")
    assert result.exit_code == 0, result.output
    assert "повільних колбеків" in result.stdout


def test_crawl_bad_url_is_one_line_error(tmp_path: Path):
    result = run(
        "crawl", "ftp://s.test/", "--out", tmp_path / "p.jsonl", "--log", tmp_path / "c.log"
    )
    assert result.exit_code == 1
    assert "помилка:" in result.output and "Traceback" not in result.output


def test_crawl_does_not_leave_logging_changes(tmp_path: Path):
    import logging

    before = (logging.getLogger("findex.crawl").level, logging.getLogger("findex.crawl").propagate)
    crawl(tmp_path)
    after = (logging.getLogger("findex.crawl").level, logging.getLogger("findex.crawl").propagate)
    assert before == after
    assert logging.getLogger("findex.crawl").handlers == []


def test_index_and_search_crawled_pages(tmp_path: Path):
    crawl(tmp_path)
    index = tmp_path / "web.bin"
    built = run("index", tmp_path / "pages.jsonl", "--out", index, "--no-memory")
    assert built.exit_code == 0, built.output
    assert "документів:   3" in built.stdout

    found = run("search", index, "whale", "--json")
    assert found.exit_code == 0, found.output
    rows = [json.loads(line) for line in found.stdout.splitlines()]
    assert [r["path"] for r in rows] == ["http://s.test/whale"]
    assert rows[0]["title"] == "Whale Book"
    assert "whale" in (rows[0]["snippet"] or "").lower()  # сніпет узято з .jsonl


def test_iter_documents_reads_jsonl_and_skips_bad_lines(tmp_path: Path):
    path = tmp_path / "pages.jsonl"
    path.write_text(
        '{"url": "http://a.test/", "title": "A", "text": "one"}\n'
        "\n"
        "{обірваний запис\n"
        '{"url": "http://b.test/", "text": "two"}\n'
        '{"url": 5, "text": "bad type"}\n'
        '{"title": "no url", "text": "x"}\n',
        encoding="utf-8",
    )
    docs = list(iter_documents(path))
    assert [(d.doc_id, d.title, d.text) for d in docs] == [
        ("http://a.test/", "A", "one"),
        ("http://b.test/", "http://b.test/", "two"),  # без title беремо url
    ]
