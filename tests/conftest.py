import logging
from pathlib import Path

import pytest
from hypothesis import settings

from helpers import make_docs

# Hypothesis: без deadline (на повільному CI перший приклад може затриматись і дати зайвий збій)
settings.register_profile("default", deadline=None)
settings.load_profile("default")

# невелика "бібліотека" у форматі Project Gutenberg: шапка з Title, маркери START/END
BOOKS = {
    "a.txt": "Title: Whale Book\n\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
    "the white whale swims in the sea\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n",
    "b.txt": "Title: Ship Book\n\n*** START OF THE PROJECT GUTENBERG EBOOK Y ***\n"
    "the ship sails the sea\n*** END OF THE PROJECT GUTENBERG EBOOK Y ***\n",
}


@pytest.fixture
def docs():
    return make_docs()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Тека з двома книгами (BOOKS) на диску."""
    root = tmp_path / "data"
    root.mkdir()
    for name, text in BOOKS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def restore_root_logger():
    """CLI налаштовує кореневий логер; після тесту повертаємо як було."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
