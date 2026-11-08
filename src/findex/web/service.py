"""Пошук для веб-шару: той самий код обслуговує і JSON API, і HTML-сторінку."""

import math
import re
import time
from collections.abc import Callable, Collection, Sequence

from markupsafe import Markup, escape

from findex.index import Index
from findex.ranking import ScorerName, make_scorer
from findex.searcher import Searcher
from findex.snippets import make_snippet
from findex.tokenizer import term_of
from findex.web.schemas import ResultItem, SearchResponse

TextOf = Callable[[int], str | None]  # doc_id -> текст документа (з кешу)

MAX_K = 50
MAX_PAGE = 100


def run_search(
    index: Index,
    searcher: Searcher,
    text_of: TextOf,
    q: str,
    k: int,
    scorer: ScorerName,
    page: int,
) -> SearchResponse:
    """Сторінка результатів. Кидає QueryError, якщо запит не розбирається.

    Щоб показати сторінку N, беремо топ N*k і відкидаємо перші (N-1)*k: рейтинг рахується
    для всіх знайдених документів у будь-якому разі, а top-k через heapq дешевий.
    """
    started = time.perf_counter()
    ranked = searcher.search(q, make_scorer(scorer), page * k)
    matched = searcher.matched(q)  # дерево й id вже в кеші після search
    terms = matched.node.terms()
    total = len(matched.doc_ids)

    items: list[ResultItem] = []
    for result in ranked[(page - 1) * k : page * k]:
        text = text_of(result.doc_id)
        snippet = None if text is None else make_snippet(text, terms)
        items.append(
            ResultItem(
                doc_id=result.doc_id,
                title=result.title,
                score=round(result.score, 4),
                path=index.doc_meta[result.doc_id].path,
                snippet=None if snippet is None else snippet.text,
                highlights=[] if snippet is None else list(snippet.spans),
            )
        )
    return SearchResponse(
        query=q,
        scorer=scorer,
        total=total,
        page=page,
        pages=max(1, math.ceil(total / k)),
        k=k,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=items,
    )


_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")


def highlight_spans(text: str, terms: Collection[str]) -> list[tuple[int, int]]:
    """Де в тексті стоять слова, що після токенізації дають один із термів запиту."""
    wanted = set(terms)
    return [m.span() for m in _WORD.finditer(text) if term_of(m.group()) in wanted]


def mark(text: str, spans: Sequence[tuple[int, int]]) -> Markup:
    """Текст для HTML: усе екранується, а збіги обгортаються в <mark>."""
    parts: list[Markup | str] = []
    last = 0
    for start, end in spans:
        parts.append(escape(text[last:start]))
        parts.append(Markup("<mark>") + escape(text[start:end]) + Markup("</mark>"))
        last = end
    parts.append(escape(text[last:]))
    return Markup("").join(parts)
