"""Застосунок FastAPI. `create_app()` це фабрика: uvicorn викликає її в кожному воркері."""

import asyncio
import functools
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from findex.searcher import Searcher
from findex.snippets import document_text
from findex.store import load
from findex.web import api, pages
from findex.web.deps import AppState
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def load_state(settings: Settings, state: AppState) -> None:
    """Читає індекс із диска й готує Searcher та кеш текстів. Кидає OSError/ValueError."""
    started = time.perf_counter()
    index = load(settings.index_path)
    if settings.corpus_root is not None:
        index.root = str(settings.corpus_root)
    state.index = index
    state.searcher = Searcher(index, cache_size=settings.cache_size)

    @functools.lru_cache(maxsize=settings.text_cache)
    def cached_text(doc_id: int) -> str | None:
        return document_text(index, doc_id)

    state.text_of = cached_text
    state.postings = index.num_postings()
    state.index_file_mb = round(settings.index_path.stat().st_size / 1024 / 1024, 1)
    log.info(
        "індекс завантажено: %s, документів %d, термів %d, %.1f с",
        settings.index_path,
        index.num_docs,
        len(index),
        time.perf_counter() - started,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        logging.getLogger("findex").setLevel(settings.log_level.upper())
        try:
            # у потоці: цикл подій не стоїть, поки читається файл на десятки мегабайт
            await asyncio.to_thread(load_state, settings, state)
        except (OSError, ValueError):
            # сервіс лишається живим, але /health дає 503: платформа не пустить на нього трафік
            log.exception("не вдалося завантажити індекс з %s", settings.index_path)
        yield
        log.info("зупинка")

    try:
        number = version("findex")
    except PackageNotFoundError:
        number = "dev"
    app = FastAPI(
        title="findex",
        version=number,
        description="Пошук по корпусу книг: інвертований індекс, BM25 і TF-IDF, мова запитів.",
        lifespan=lifespan,
    )
    app.state.findex = state

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # стек іде в лог разом з id, а клієнт бачить лише id, за яким цей запис знайти
            log.exception(
                "необроблена помилка rid=%s %s %s", request_id, request.method, request.url
            )
            response = JSONResponse(
                {"detail": "внутрішня помилка сервера", "request_id": request_id}, status_code=500
            )
        response.headers["X-Request-ID"] = request_id
        log.info(
            "%s %s -> %d, %.1f мс rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return response

    app.include_router(api.router)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
