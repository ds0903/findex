"""Сніпети: уривок тексту навколо найкращого збігу, з виділеними термами."""

import json
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from findex.documents import strip_boilerplate
from findex.index import Index
from findex.tokenizer import term_of

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Snippet:
    text: str
    spans: tuple[tuple[int, int], ...]  # де в text стоять терми запиту: (початок, кінець)

    def highlight(self, start: str = "**", end: str = "**") -> str:
        """text, де кожен збіг обгорнуто в start/end."""
        out: list[str] = []
        last = 0
        for a, b in self.spans:
            out.append(self.text[last:a])
            out.append(f"{start}{self.text[a:b]}{end}")
            last = b
        out.append(self.text[last:])
        return "".join(out)


# символи, які re.IGNORECASE вважає тим самим, що й ASCII-літера (K, ſ, İ, ı): для текстів
# з ними швидкий шлях не годиться, бо str.lower() їх з літерами не зіставляє
_FOLDS_TO_ASCII = "Kſİı"


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"  # те саме, що \w у re


def _find_matches(text: str, terms: Collection[str]) -> list[tuple[int, int, str]]:
    """(початок, кінець, терм) усіх входжень термів у тексті, за зростанням початку.

    Профіль лаби 8: майже весь час пошуку йшов у регулярному виразі з (?<!\\w) на кожній позиції
    книги на ≈1.5 млн символів. Швидкий шлях шукає самі літерали в тексті з малими літерами (рушій
    `re` тоді має префільтр) і перевіряє межі слів вручну; результат той самий (є тест).
    Для термів з дефісом чи апострофом і для екзотичних символів лишається старий вираз.
    """
    if not terms:
        return []
    simple = all(t.isascii() and t.isalnum() for t in terms)
    if simple and not any(ch in text for ch in _FOLDS_TO_ASCII):
        lowered = text.lower()
        if len(lowered) == len(text):
            return _find_literals(text, lowered, terms)
    return _find_with_regex(text, terms)


def _find_literals(text: str, lowered: str, terms: Collection[str]) -> list[tuple[int, int, str]]:
    pattern = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)))
    found: list[tuple[int, int, str]] = []
    for m in pattern.finditer(lowered):
        start, end = m.span()
        if (start and _is_word_char(text[start - 1])) or (
            end < len(text) and _is_word_char(text[end])
        ):
            continue
        term = term_of(text[start:end])
        if term in terms:
            found.append((start, end, term))
    return found


def _find_with_regex(text: str, terms: Collection[str]) -> list[tuple[int, int, str]]:
    """Повільний, але універсальний варіант: один вираз із межами слів."""
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
        + r")(?!\w)",
        re.IGNORECASE,
    )
    found: list[tuple[int, int, str]] = []
    for m in pattern.finditer(text):
        term = term_of(m.group())
        if term in terms:
            found.append((m.start(), m.end(), term))
    return found


def _best_window(matches: list[tuple[int, int, str]], width: int) -> int:
    """Індекс збігу, навколо якого (±width символів) вміщається найбільше різних термів,
    а при рівності - найбільше збігів. Двома вказівниками за один прохід."""
    counts: Counter[str] = Counter()
    left = right = 0
    best, best_score = 0, (-1, -1)
    for i, (start, _, _) in enumerate(matches):
        while right < len(matches) and matches[right][0] <= start + width:
            counts[matches[right][2]] += 1
            right += 1
        while matches[left][0] < start - width:
            term = matches[left][2]
            counts[term] -= 1
            if counts[term] == 0:
                del counts[term]
            left += 1
        score = (len(counts), right - left)
        if score > best_score:
            best, best_score = i, score
    return best


def make_snippet(text: str, terms: Collection[str], width: int = 80) -> Snippet | None:
    """Вікно ±width символів навколо найкращого входження термів; None, якщо збігів нема."""
    if not terms:
        return None
    text = unicodedata.normalize("NFC", text)
    wanted = set(terms)
    matches = _find_matches(text, wanted)
    if not matches:
        return None

    start, end, _ = matches[_best_window(matches, width)]
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    chunk = text[lo:hi]
    # не лишаємо обрізаних слів по краях
    if lo > 0 and text[lo - 1].isalnum():
        chunk = re.sub(r"^\w+", "", chunk, count=1)
    if hi < len(text) and text[hi].isalnum():
        chunk = re.sub(r"\w+$", "", chunk, count=1)
    chunk = " ".join(chunk.split())  # переноси рядків -> пробіли

    prefix = "… " if lo > 0 else ""
    suffix = " …" if hi < len(text) else ""
    spans = tuple((a + len(prefix), b + len(prefix)) for a, b, _ in _find_matches(chunk, wanted))
    return Snippet(prefix + chunk + suffix, spans)


def document_text(index: Index, doc_id: int) -> str | None:
    """Текст документа з диска (індекс зберігає лише шлях). None, якщо файл не прочитався."""
    meta_path = index.doc_meta[doc_id].path
    if Path(index.root).suffix == ".jsonl":  # сторінка краулера: шлях це url у .jsonl
        return _text_from_jsonl(Path(index.root), meta_path)
    path = Path(index.root) / meta_path
    try:
        return strip_boilerplate(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        log.warning("не можу прочитати %s для сніпета: %s", path, exc)
        return None


def _text_from_jsonl(path: Path, url: str) -> str | None:
    """Шукає сторінку за url у файлі краулера. JSON розбирається лише для стрічки з цим url."""
    needle = json.dumps(url, ensure_ascii=False)
    try:
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if needle in line:
                    record = json.loads(line)
                    if record.get("url") == url:
                        return str(record["text"])
    except (OSError, ValueError, KeyError) as exc:
        log.warning("не можу прочитати %s для сніпета: %s", path, exc)
    return None


def snippet_for(
    index: Index, doc_id: int, terms: Collection[str], width: int = 80
) -> Snippet | None:
    text = document_text(index, doc_id)
    return None if text is None else make_snippet(text, terms, width)
