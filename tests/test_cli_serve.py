import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Підміняє uvicorn.run: перевіряємо, з чим команда його викликає, а сервер не піднімаємо."""
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda target, **kwargs: calls.append(kwargs | {"app": target})
    )
    for name in ("INDEX_PATH", "CORPUS_ROOT", "PORT", "WEB_CONCURRENCY", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    return calls


def test_serve_defaults(uvicorn_calls: list[dict[str, Any]]):
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    (call,) = uvicorn_calls
    assert call["app"] == "findex.web.app:create_app" and call["factory"] is True
    assert (call["host"], call["port"], call["workers"]) == ("127.0.0.1", 8000, 1)


def test_serve_options_reach_workers_through_environment(
    uvicorn_calls: list[dict[str, Any]], tmp_path: Path
):
    index = tmp_path / "i.bin"
    result = runner.invoke(
        app,
        ["serve", "--index", str(index), "--port", "9001", "--workers", "4", "--host", "0.0.0.0"],
    )
    assert result.exit_code == 0, result.output
    (call,) = uvicorn_calls
    assert (call["host"], call["port"], call["workers"]) == ("0.0.0.0", 9001, 4)
    # воркери uvicorn стартують окремими процесами й читають Settings саме з оточення
    assert os.environ["INDEX_PATH"] == str(index.resolve())
    os.environ.pop("INDEX_PATH")


def test_serve_reads_platform_variables(
    uvicorn_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    runner.invoke(app, ["serve"])
    (call,) = uvicorn_calls
    assert (call["port"], call["workers"]) == (10000, 2)


def test_serve_bad_port_variable_is_a_one_line_error(
    uvicorn_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PORT", "abc")
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1 and "помилка:" in result.output
    assert uvicorn_calls == []
