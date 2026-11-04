import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# маркери початку/кінця самої книги в файлах Gutenberg (решта - ліцензійний текст)
_START = re.compile(r"^\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)
_END = re.compile(r"^\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)
_TITLE = re.compile(r"^Title:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: Path
    text: str
    title: str = ""


def extract_title(raw: str, default: str = "") -> str:
    """Бере назву з рядка `Title: ...` у шапці Gutenberg. Якщо його нема - default."""
    match = _TITLE.search(raw[:5000])
    return match.group(1) if match else default


def strip_boilerplate(text: str) -> str:
    """Відрізає шапку й ліцензію Gutenberg. Якщо маркерів нема - повертає текст як є."""
    start = _START.search(text)
    if start:
        text = text[start.end() :]
    end = _END.search(text)
    if end:
        text = text[: end.start()]
    return text


def iter_documents(
    root: Path | str, pattern: str = "*.txt", strip: bool = True
) -> Iterator[Document]:
    """Читає документи з root по одному.

    Це генератор: файл відкривається тільки коли в нього дійшла черга, тож у пам'яті
    одночасно живе один документ. Список шляхів (не текстів) збирається наперед
    заради стабільного порядку - це кілька сотень рядків, не мегабайти.

    Файли, які не вдалося прочитати (не utf-8, немає доступу), пропускаються
    з записом у лог.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"немає такої теки: {root}")

    for path in sorted(root.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("пропускаю %s: %s", path, exc)
            continue
        title = extract_title(text, default=path.stem)  # до того як відріжемо шапку
        if strip:
            text = strip_boilerplate(text)
        yield Document(doc_id=path.relative_to(root).as_posix(), path=path, text=text, title=title)
