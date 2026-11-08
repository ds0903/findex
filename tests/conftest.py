import logging

import pytest

from helpers import make_docs


@pytest.fixture
def docs():
    return make_docs()


@pytest.fixture(autouse=True)
def restore_root_logger():
    """CLI налаштовує кореневий логер; після тесту повертаємо як було."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
