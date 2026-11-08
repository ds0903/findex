"""Залежності (Depends). Тести підміняють їх через app.dependency_overrides, тож
справжній файл індексу тестам не потрібен."""

import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from findex.index import Index
from findex.searcher import Searcher
from findex.snippets import document_text
from findex.web.service import TextOf
from findex.web.settings import Settings
from findex.web.settings import get_settings as _get_settings


@dataclass
class AppState:
    """Усе, що lifespan завантажує один раз при старті."""

    index: Index | None = None
    searcher: Searcher | None = None
    text_of: TextOf | None = None  # текст документа з LRU-кешем
    postings: int = 0  # num_postings() обходить увесь словник, тому рахуємо один раз
    index_file_mb: float | None = None
    started: float = field(default_factory=time.monotonic)


def get_settings() -> Settings:
    return _get_settings()


def get_state(request: Request) -> AppState:
    return request.app.state.findex


def get_index(state: Annotated[AppState, Depends(get_state)]) -> Index:
    if state.index is None:
        raise HTTPException(status_code=503, detail="індекс ще не завантажений")
    return state.index


def get_searcher(state: Annotated[AppState, Depends(get_state)]) -> Searcher:
    if state.searcher is None:
        raise HTTPException(status_code=503, detail="індекс ще не завантажений")
    return state.searcher


def plain_text_of(index: Index) -> TextOf:
    return lambda doc_id: document_text(index, doc_id)  # без кешу


def get_text_of(
    state: Annotated[AppState, Depends(get_state)], index: Annotated[Index, Depends(get_index)]
) -> TextOf:
    return state.text_of if state.text_of is not None else plain_text_of(index)
