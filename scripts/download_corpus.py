"""Завантажує книги з Project Gutenberg в data/gutenberg/.

Запуск: uv run python scripts/download_corpus.py
Файли, які вже є на диску, пропускаються.
"""

import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# id книг на gutenberg.org (переважно англійською + кілька іншими мовами для Unicode)
_ENGLISH = """
1342 11 84 1661 2701 98 1232 174 345 76 74 1400 2600 1184 135 996 2554 1399 4300
158 161 1260 768 120 35 36 43 16 55 514 215 244 2542 1727 6130 2097 1998 2591 1497
3207 205 408 829 730 766 580 1023 786 1524 100
"""
# не англійською (для перевірки Unicode)
_OTHER = "17489 2229 7205 5200 19942 30774 46853 12816 8800"

BOOK_IDS = [int(x) for x in (_ENGLISH + _OTHER).split()]

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gutenberg"


def download(book_id: int) -> Path | None:
    target = OUT_DIR / f"pg{book_id}.txt"
    if target.exists():
        return target
    req = urllib.request.Request(
        URL.format(id=book_id), headers={"User-Agent": "findex-lab/0.1 (student project)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  {book_id}: пропущено ({exc})")
        return None
    target.write_bytes(data)
    return target


def title_and_language(path: Path) -> tuple[str, str]:
    head = path.read_text(encoding="utf-8-sig", errors="replace")[:3000]
    title = re.search(r"^Title:\s*(.+)$", head, re.MULTILINE)
    lang = re.search(r"^Language:\s*(.+)$", head, re.MULTILINE)
    return (
        title.group(1).strip() if title else "?",
        lang.group(1).strip() if lang else "?",
    )


def main() -> int:
    # консоль windows часто в cp1251, а в назвах книг є é, ö тощо
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for book_id in BOOK_IDS:
        path = download(book_id)
        if path is None:
            continue
        title, lang = title_and_language(path)
        size = path.stat().st_size
        total += size
        print(f"{book_id:>6}  {size / 1024:>7.0f} KB  {lang:<10} {title[:60]}")
        time.sleep(0.5)  # не навантажуємо сервер
    print(f"\nрозмір корпусу: {total / 1024 / 1024:.1f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
