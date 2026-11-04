import random

import pytest

from findex.index import STORAGES, build_index
from findex.search import (
    ENGINES,
    merge_and,
    merge_diff,
    merge_or,
    search,
    set_and,
    set_diff,
    set_or,
)
from helpers import make_docs

# doc 0: the cat sat on the mat
# doc 1: the dog sat
# doc 2: a cat and a dog

QUERIES = [
    ("cat", [0, 2]),
    ("cat dog", [2]),  # неявний AND
    ("cat AND dog", [2]),
    ("cat OR dog", [0, 1, 2]),
    ("cat NOT dog", [0]),
    ("NOT cat", [1]),
    ("the OR dog NOT sat", [2]),  # зліва направо: (the OR dog) NOT sat
    ("a OR NOT cat", [1, 2]),
    ("sat NOT the", []),
    ("CAT Dog", [2]),  # терми нормалізуються як і в токенізаторі
    ("cat OR", [0, 2]),  # операція без правого операнда ігнорується
    ("unknown", []),
    ("cat unknown", []),
    ("cat OR unknown", [0, 2]),
    ("", []),
    ("   ", []),
    ("OR AND NOT", []),
    ("cat or dog", []),  # оператори тільки великими літерами, "or" тут звичайний терм
]


@pytest.fixture(params=STORAGES)
def index(request):
    return build_index(make_docs(), storage=request.param)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("query", "expected"), QUERIES)
def test_queries(index, engine, query, expected):
    assert search(index, query, engine) == expected


def test_unknown_engine(index):
    with pytest.raises(ValueError):
        search(index, "cat", engine="magic")


def test_search_does_not_modify_index(index):
    before = {t: index.df(t) for t in index.postings}
    search(index, "cat unknown NOT zzz", "merge")
    assert {t: index.df(t) for t in index.postings} == before
    assert "zzz" not in index.postings


def test_result_is_a_list_not_the_internal_array():
    index = build_index(make_docs(), storage="array")
    result = search(index, "cat")
    assert isinstance(result, list)
    result.append(99)
    assert list(index.postings["cat"][0]) == [0, 2]  # внутрішній масив не зачепило


def test_merge_matches_python_sets_on_random_lists():
    rng = random.Random(1)
    for _ in range(300):
        a = sorted(rng.sample(range(60), rng.randint(0, 30)))
        b = sorted(rng.sample(range(60), rng.randint(0, 30)))
        assert merge_and(a, b) == sorted(set(a) & set(b)) == set_and(a, b)
        assert merge_or(a, b) == sorted(set(a) | set(b)) == set_or(a, b)
        assert merge_diff(a, b) == sorted(set(a) - set(b)) == set_diff(a, b)


def test_merge_edge_cases():
    assert merge_and([], [1, 2]) == []
    assert merge_or([], [1, 2]) == [1, 2]
    assert merge_or([1, 2], []) == [1, 2]
    assert merge_diff([1, 2], []) == [1, 2]
    assert merge_diff([], [1]) == []
    assert merge_and([1, 3, 5], [2, 4, 6]) == []
    assert merge_or([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
