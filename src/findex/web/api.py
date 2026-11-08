"""JSON API: /search, /docs/{doc_id}, /stats, /health."""

import time
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from findex.index import Index
from findex.query import QueryError
from findex.ranking import ScorerName
from findex.searcher import Searcher
from findex.web.deps import AppState, get_index, get_searcher, get_state, get_text_of
from findex.web.schemas import (
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    SearchResponse,
    StatsResponse,
)
from findex.web.service import MAX_K, MAX_PAGE, TextOf, run_search

router = APIRouter()

Q = Annotated[str, Query(min_length=1, max_length=200, description='запит: whale AND "white sea"')]
K = Annotated[int, Query(ge=1, le=MAX_K, description="результатів на сторінці")]
Page = Annotated[int, Query(ge=1, le=MAX_PAGE, description="номер сторінки")]


@router.get("/search", response_model=SearchResponse, responses={422: {"model": ErrorResponse}})
def search(
    q: Q,
    index: Annotated[Index, Depends(get_index)],
    searcher: Annotated[Searcher, Depends(get_searcher)],
    text_of: Annotated[TextOf, Depends(get_text_of)],
    k: K = 10,
    scorer: ScorerName = "bm25",
    page: Page = 1,
) -> SearchResponse:
    """Ранжований пошук із сніпетами.

    Звичайний `def`, а не `async def`: пошук це чисто процесорна робота (десятки мс), і в
    `async def` вона зайняла б цикл подій, поки не скінчиться. FastAPI виконує `def`-маршрут
    у пулі потоків, тож цикл далі приймає з'єднання й віддає /health.
    """
    try:
        return run_search(index, searcher, text_of, q, k, scorer, page)
    except QueryError as exc:
        raise HTTPException(status_code=422, detail=f"некоректний запит: {exc}") from exc


@router.get(
    "/docs/{doc_id}", response_model=DocumentResponse, responses={404: {"model": ErrorResponse}}
)
def get_document(
    doc_id: int,
    index: Annotated[Index, Depends(get_index)],
    text_of: Annotated[TextOf, Depends(get_text_of)],
    limit: Annotated[
        int, Query(ge=0, le=2_000_000, description="скільки символів тексту")
    ] = 20_000,
) -> DocumentResponse:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"документа {doc_id} нема")
    text = text_of(doc_id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"текст документа {doc_id} недоступний")
    return DocumentResponse(
        doc_id=doc_id,
        title=meta.title,
        path=meta.path,
        tokens=index.doc_length(doc_id),
        text=text[:limit],
        truncated=len(text) > limit,
    )


@router.get("/stats", response_model=StatsResponse)
def stats(
    index: Annotated[Index, Depends(get_index)],
    state: Annotated[AppState, Depends(get_state)],
) -> StatsResponse:
    try:
        number = version("findex")
    except PackageNotFoundError:
        number = "dev"
    return StatsResponse(
        documents=index.num_docs,
        terms=len(index),
        postings=state.postings,
        avg_doc_length=round(index.avg_doc_length, 1),
        storage=index.storage,
        index_file_mb=state.index_file_mb,
        uptime_seconds=round(time.monotonic() - state.started, 1),
        version=number,
    )


@router.get("/health", response_model=HealthResponse, responses={503: {"model": ErrorResponse}})
def health(index: Annotated[Index, Depends(get_index)]) -> HealthResponse:
    """200, якщо індекс завантажений; 503, поки ні (платформа не шле трафік на такий екземпляр)."""
    return HealthResponse(status="ok", documents=index.num_docs)
