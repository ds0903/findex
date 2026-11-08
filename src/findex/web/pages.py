"""HTML-сторінки: пошук (/) і перегляд документа (/view/{doc_id}). Сторінки Jinja2, без JS."""

import math
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from findex.index import Index
from findex.query import QueryError, parse
from findex.ranking import SCORER_NAMES, ScorerName
from findex.web.deps import AppState, get_state, get_text_of, plain_text_of
from findex.web.service import MAX_K, MAX_PAGE, TextOf, highlight_spans, mark, run_search

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.globals["mark"] = mark  # pyright: ignore[reportUnknownMemberType]

PART_CHARS = 6000  # скільки символів документа показуємо за раз


def _page_window(page: int, pages: int, width: int = 2) -> list[int]:
    """Номери сторінок навколо поточної: 1 ... 4 5 [6] 7 8 ... 20."""
    wanted = {1, pages, *range(max(1, page - width), min(pages, page + width) + 1)}
    return sorted(wanted)


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    state: Annotated[AppState, Depends(get_state)],
    q: str = "",
    scorer: str = "bm25",
    k: int = 10,
    page: int = 1,
) -> HTMLResponse:
    k = min(max(k, 1), MAX_K)  # у формі краще виправити число, ніж показати JSON з помилкою 422
    page = min(max(page, 1), MAX_PAGE)
    ranking: ScorerName = scorer if scorer in SCORER_NAMES else "bm25"  # type: ignore[assignment]
    q = q.strip()[:200]

    response = None
    error = None
    ready = state.index is not None and state.searcher is not None
    if q and ready:
        assert state.index is not None and state.searcher is not None
        text_of = state.text_of or plain_text_of(state.index)
        try:
            response = run_search(state.index, state.searcher, text_of, q, k, ranking, page)
        except QueryError as exc:
            error = f"Некоректний запит: {exc}"
    context = {
        "q": q,
        "scorer": ranking,
        "scorers": SCORER_NAMES,
        "k": k,
        "response": response,
        "error": error,
        "ready": ready,
        "documents": state.index.num_docs if state.index else 0,
        "window": _page_window(response.page, response.pages) if response else [],
    }
    return templates.TemplateResponse(request, "index.html", context, status_code=200)


@router.get("/view/{doc_id}", response_class=HTMLResponse)
def view_document(
    request: Request,
    doc_id: int,
    state: Annotated[AppState, Depends(get_state)],
    text_of: Annotated[TextOf, Depends(get_text_of)],
    q: str = "",
    part: Annotated[int, Query(ge=1)] = 1,
) -> HTMLResponse:
    index: Index | None = state.index
    if index is None or doc_id not in index.doc_meta:
        raise HTTPException(status_code=404, detail=f"документа {doc_id} нема")
    text = text_of(doc_id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"текст документа {doc_id} недоступний")

    parts = max(1, math.ceil(len(text) / PART_CHARS))
    part = min(part, parts)
    chunk = text[(part - 1) * PART_CHARS : part * PART_CHARS]
    spans: list[tuple[int, int]] = []
    if q.strip():
        try:
            spans = highlight_spans(chunk, parse(q).terms())
        except QueryError:
            spans = []  # поганий запит не заважає читати документ
    meta = index.doc_meta[doc_id]
    context = {
        "doc_id": doc_id,
        "title": meta.title,
        "path": meta.path,
        "tokens": index.doc_length(doc_id),
        "chunk": chunk,
        "spans": spans,
        "part": part,
        "parts": parts,
        "q": q.strip(),
    }
    return templates.TemplateResponse(request, "view.html", context)
