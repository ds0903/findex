# syntax=docker/dockerfile:1

# --- етап 1: збірка. Тут є uv, компілятор залежностей і мережа для корпусу ---
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# спершу лише залежності: шар кешується, доки не зміняться pyproject.toml чи uv.lock
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --no-editable

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# корпус і індекс будуються під час збірки образу, тож у git лежить лише код.
# Інший індекс можна підставити під час запуску: -e INDEX_PATH=... -v ...
COPY scripts/download_corpus.py scripts/
ARG INDEX_FLAGS="--positions"
RUN .venv/bin/python scripts/download_corpus.py \
    && .venv/bin/findex index data/gutenberg --out data/index.bin --no-memory ${INDEX_FLAGS}

# --- етап 2: те, що запускається. Без uv, без компіляторів, без кешів ---
FROM python:3.13-slim
RUN useradd --create-home --uid 10001 findex
WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/data /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/data/index.bin \
    CORPUS_ROOT=/app/data/gutenberg \
    PORT=8000

USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD ["python", "-c", "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))"]

# порт береться зі змінної PORT (її виставляє платформа), воркери зі змінної WEB_CONCURRENCY
CMD ["findex", "serve", "--host", "0.0.0.0"]
