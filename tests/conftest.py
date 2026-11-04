import pytest

from helpers import make_docs


@pytest.fixture
def docs():
    return make_docs()
