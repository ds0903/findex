"""Локальне дзеркало для перевірки краулера (лаба 6): сайт із книг корпусу на 127.0.0.1.

Це "свій сайт": його можна обходити без чужого дозволу й без навантаження на чужі сервери.
Кожна книга стає сторінкою зі списком частин, кожна частина це кусок тексту з посиланнями
вперед, назад і на інші книги. Відповідь має штучну затримку (як у справжнього сайту),
інакше локальний сервер відповідав би швидше, ніж краулер встиг би щось одночасно тримати.

    python scripts/serve_site.py data/gutenberg --port 8765 --latency 0.08 --faults light

--faults none   лише нормальні сторінки
--faults light  + /fault/flaky (429 з Retry-After один раз), /fault/missing (404),
                  /fault/loop (редірект сам на себе)
--faults all    + /fault/slow (відповідь через 5 с, довше за таймаут) і /fault/error (завжди 503)
GET /__stats    скільки запитів прийнято й скільки було одночасно (перевірка з боку сервера)
"""

import argparse
import hashlib
import html
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from findex.documents import extract_title, strip_boilerplate

WORDS_PER_PART = 450


class Site:
    def __init__(self, corpus: Path, parts: int, latency: float, faults: str) -> None:
        self.latency = latency
        self.faults = faults
        self.flaky_served: set[str] = set()
        self.lock = threading.Lock()
        self.requests = 0
        self.active = 0
        self.max_active = 0
        self.books: dict[str, tuple[str, list[str]]] = {}
        for path in sorted(corpus.glob("*.txt")):
            raw = path.read_text(encoding="utf-8-sig")
            words = strip_boilerplate(raw).split()[1500:]  # без змісту на початку
            chunks = [
                " ".join(words[i * WORDS_PER_PART : (i + 1) * WORDS_PER_PART]) for i in range(parts)
            ]
            self.books[path.stem] = (extract_title(raw, path.stem), [c for c in chunks if c])

    def delay_for(self, path: str) -> float:
        h = int(hashlib.md5(path.encode()).hexdigest()[:4], 16) / 0xFFFF  # 0..1, стабільно
        return self.latency * (0.7 + 0.6 * h)

    def page(self, title: str, body: str) -> bytes:
        text = f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(title)}</title>"
        text += f"</head><body><h1>{html.escape(title)}</h1>{body}</body></html>"
        return text.encode()

    def render(self, path: str) -> tuple[int, dict[str, str], bytes]:
        slugs = list(self.books)
        if path == "/":
            links = "".join(
                f'<li><a href="/book/{s}">{html.escape(self.books[s][0])}</a></li>' for s in slugs
            )
            links += '<li><a href="/private/secret">закрита сторінка</a></li>'
            if self.faults != "none":
                links += "".join(
                    f'<li><a href="/fault/{name}">{name}</a></li>'
                    for name in ("flaky", "missing", "loop")
                )
            if self.faults == "all":
                links += (
                    '<li><a href="/fault/slow">slow</a></li><li><a href="/fault/error">e</a></li>'
                )
            return 200, {}, self.page("Бібліотека", f"<ul>{links}</ul>")

        parts = path.strip("/").split("/")
        if parts[0] == "book" and len(parts) in (2, 3) and parts[1] in self.books:
            slug = parts[1]
            title, chunks = self.books[slug]
            if len(parts) == 2:
                items = "".join(
                    f'<li><a href="/book/{slug}/{n}">частина {n}</a></li>'
                    for n in range(1, len(chunks) + 1)
                )
                return 200, {}, self.page(title, f'<ul>{items}</ul><a href="/">на головну</a>')
            if parts[2].isdigit() and 1 <= int(parts[2]) <= len(chunks):
                n = int(parts[2])
                i = slugs.index(slug)
                nav = f'<a href="/book/{slug}/">до книги</a> <a href="/book/{slug}#top">зміст</a> '
                if n > 1:
                    nav += f'<a href="/book/{slug}/{n - 1}">назад</a> '
                if n < len(chunks):
                    nav += f'<a href="/book/{slug}/{n + 1}">далі</a> '
                other = slugs[(i + n) % len(slugs)]  # перехресне посилання на іншу книгу
                nav += f'<a href="/book/{other}/1">схоже</a>'
                return (
                    200,
                    {},
                    self.page(f"{title}, частина {n}", f"<p>{html.escape(chunks[n - 1])}</p>{nav}"),
                )

        if path == "/robots.txt":
            return 200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /private/\n"
        if path == "/private/secret":
            return 200, {}, self.page("Закрито", "<p>сюди краулер заходити не повинен</p>")
        if path == "/fault/flaky":
            with self.lock:
                first = path not in self.flaky_served
                self.flaky_served.add(path)
            if first:
                return 429, {"Retry-After": "1"}, b""
            return 200, {}, self.page("Після 429", "<p>вдалося з другого разу</p>")
        if path == "/fault/loop":
            return 302, {"Location": "/fault/loop"}, b""
        if path == "/fault/error":
            return 503, {}, b""
        if path == "/fault/slow":
            time.sleep(5)
            return 200, {}, self.page("Повільна", "<p>запізно</p>")
        return 404, {}, b""


def make_handler(site: Site) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, як у справжніх серверів

        def do_GET(self) -> None:
            path = self.path.split("?")[0].split("#")[0]
            if path == "/__stats":
                body = json.dumps({"requests": site.requests, "max_active": site.max_active})
                return self.reply(200, {"Content-Type": "application/json"}, body.encode())
            with site.lock:
                site.requests += 1
                site.active += 1
                site.max_active = max(site.max_active, site.active)
            try:
                if path != "/robots.txt" and not path.startswith("/fault/slow"):
                    time.sleep(site.delay_for(path))
                status, headers, body = site.render(path)
            finally:
                with site.lock:
                    site.active -= 1
            self.reply(status, headers, body)

        def reply(self, status: int, headers: dict[str, str], body: bytes) -> None:
            self.send_response(status)
            headers.setdefault("Content-Type", "text/html; charset=utf-8")
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--latency", type=float, default=0.08, help="секунд на відповідь")
    parser.add_argument("--parts", type=int, default=6, help="частин на книгу")
    parser.add_argument("--faults", choices=("none", "light", "all"), default="light")
    args = parser.parse_args()

    site = Site(args.corpus, args.parts, args.latency, args.faults)
    pages = 1 + sum(1 + len(chunks) for _, chunks in site.books.values())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(site))
    server.daemon_threads = True
    print(
        f"сайт на http://127.0.0.1:{args.port}/ ({len(site.books)} книг, ~{pages} сторінок)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
