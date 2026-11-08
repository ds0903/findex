"""Розбір HTML: заголовок, видимий текст і посилання (html.parser зі стандартної бібліотеки)."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class ParsedPage:
    title: str
    text: str
    links: tuple[str, ...]
    noindex: bool = False  # <meta name="robots" content="noindex">
    nofollow: bool = False


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.directives: set[str] = set()
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"] or "")
        elif tag == "meta" and (attributes.get("name") or "").lower() == "robots":
            content = (attributes.get("content") or "").lower()
            self.directives.update(part.strip() for part in content.split(","))
        elif tag == "title":
            self._in_title = True
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.text_parts.append(data)


def parse_html(html: str) -> ParsedPage:
    collector = _Collector()
    collector.feed(html)
    collector.close()
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(collector.text_parts))
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return ParsedPage(
        title=" ".join("".join(collector.title_parts).split()),
        text=text,
        links=tuple(collector.links),
        noindex="noindex" in collector.directives or "none" in collector.directives,
        nofollow="nofollow" in collector.directives or "none" in collector.directives,
    )
