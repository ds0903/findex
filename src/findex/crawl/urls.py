"""Адреси: канонічний вигляд, щоб дві адреси однієї сторінки давали один рядок."""

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}

# не сторінки: їх не качаємо (решту відсіює перевірка Content-Type)
_SKIP_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".mp3", ".mp4", ".css", ".js", ".json", ".xml",
)  # fmt: skip


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Приводить адресу до одного вигляду. None, якщо це не http(s)-адреса.

    - схема й хост у нижньому регістрі, типовий порт (80/443) прибирається;
    - фрагмент після # відкидається;
    - порожній шлях це "/", кінцевий слеш прибирається (/a/ і /a це одна сторінка);
    - параметри query сортуються (?b=2&a=1 і ?a=1&b=2 однакові).
    Відносні посилання розв'язуються відносно base.
    """
    if base is not None:
        url = urljoin(base, url)
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:  # "http://[::1" чи порт не число
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None

    netloc = f"[{host}]" if ":" in host else host  # ipv6
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        netloc += f":{port}"
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    """Хост разом з портом: за ним рахуються ліміти (localhost:8000 і localhost:9000 різні)."""
    return urlsplit(url).netloc


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def looks_like_page(url: str) -> bool:
    return not urlsplit(url).path.lower().endswith(_SKIP_SUFFIXES)
