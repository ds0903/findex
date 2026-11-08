import logging

import pytest

from findex.timing import timed


def test_wraps_keeps_name_and_docstring():
    @timed
    def search():
        """Find documents."""

    assert search.__name__ == "search"
    assert search.__doc__ == "Find documents."


def test_without_wraps_the_name_would_be_wrapper():
    # те, від чого functools.wraps рятує: проста обгортка губить ім'я
    def raw_timed(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @raw_timed
    def search():
        """Find documents."""

    assert search.__name__ == "wrapper"
    assert search.__doc__ is None


def test_passes_arguments_and_result_through():
    @timed
    def add(a, b=0):
        return a + b

    assert add(1, b=2) == 3


def test_logs_elapsed_ms_at_debug(caplog):
    @timed
    def work():
        return 1

    with caplog.at_level(logging.DEBUG):
        work()
    assert any("work:" in r.message and "мс" in r.message for r in caplog.records)
    assert all(r.levelno == logging.DEBUG for r in caplog.records)


def test_nothing_logged_at_info_level(caplog):
    @timed
    def work():
        return 1

    with caplog.at_level(logging.INFO):
        work()
    assert caplog.records == []


def test_logs_even_when_function_raises(caplog):
    @timed
    def boom():
        raise RuntimeError("x")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            boom()
    assert "boom:" in caplog.text
