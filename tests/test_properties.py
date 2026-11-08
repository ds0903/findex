"""Властивості, які мають виконуватись для будь-яких вхідних даних (Hypothesis)."""

import math
import uuid
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from findex.documents import Document
from findex.index import STORAGES, Posting, Storage, build_index
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.ranking import BM25, TfIdf, top_k
from findex.search import merge_and, merge_diff, merge_or, set_and, set_diff, set_or
from findex.store import FORMATS, Format, load, save
from findex.tokenizer import tokenize
from helpers import make_docs

# --- токенізатор ---


@given(st.text())
def test_tokenize_never_crashes_and_gives_clean_tokens(text: str):
    for token in tokenize(text):
        assert token  # жодних порожніх токенів
        assert token == token.casefold()  # вже у нижньому регістрі
        assert not token.startswith(("'", "-")) and not token.endswith(("'", "-"))


@given(st.text())
def test_tokenize_is_idempotent_on_its_own_output(text: str):
    # токени, склеєні пробілами й розібрані знову, дають ті самі токени
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


# --- булеві операції: merge дає те саме, що множини ---

sorted_ids = st.lists(st.integers(0, 300), unique=True).map(sorted)


@given(sorted_ids, sorted_ids)
def test_merge_equals_set_operations(a: list[int], b: list[int]):
    assert merge_and(a, b) == set_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == set_or(a, b) == sorted(set(a) | set(b))
    assert merge_diff(a, b) == set_diff(a, b) == sorted(set(a) - set(b))


@pytest.mark.slow
@settings(max_examples=200)
@given(
    st.lists(st.integers(0, 100_000), unique=True, max_size=3000).map(sorted),
    st.lists(st.integers(0, 100_000), unique=True, max_size=3000).map(sorted),
)
def test_merge_equals_set_operations_on_long_lists(a: list[int], b: list[int]):
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_diff(a, b) == sorted(set(a) - set(b))


# --- збереження: що записали, те й прочитали ---

words = st.sampled_from(["cat", "dog", "sat", "mat", "straße", "кіт", "naïve", "a", "the"])
texts = st.lists(words, max_size=12).map(" ".join)


def corpus_of(texts_: list[str]):
    return (
        Document(f"d{i}.txt", Path(f"d{i}.txt"), text, f"Doc {i}") for i, text in enumerate(texts_)
    )


@pytest.mark.slow
@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(texts, max_size=6),
    st.sampled_from(STORAGES),
    st.booleans(),
    st.sampled_from(FORMATS),
)
def test_save_then_load_gives_equal_index(
    tmp_path: Path, corpus_texts: list[str], storage: Storage, positions: bool, fmt: Format
):
    if storage == "array" and positions:
        positions = False  # array-сховище позицій не підтримує
    index = build_index(corpus_of(corpus_texts), storage, positions)
    path = tmp_path / f"{uuid.uuid4().hex}.idx"
    save(index, path, fmt)
    assert load(path) == index


# --- парсер: str(дерево) розбирається назад у те саме дерево ---

leaf = st.one_of(
    st.sampled_from(["a", "b", "cat", "dog", "or", "and", "not"]).map(Term),
    st.lists(st.sampled_from(["event", "loop", "the", "x"]), min_size=2, max_size=3).map(
        lambda ws: Phrase(tuple(ws))
    ),
)
trees = st.recursive(
    leaf,
    lambda children: st.one_of(
        st.builds(And, children, children),
        st.builds(Or, children, children),
        st.builds(Not, children),
    ),
    max_leaves=8,
)


@given(trees)
def test_str_then_parse_round_trips(tree: Node):
    assert parse(str(tree)) == tree


@given(trees)
def test_equal_trees_have_equal_hashes(tree: Node):
    assert hash(parse(str(tree))) == hash(tree)


# --- рейтинг ---

index_for_scores = build_index(
    make_docs(["x y z", "x " * 30 + "y", "z z z z", "w"]),
)


@given(st.integers(1, 500), st.sampled_from([0, 1, 2, 3]))
def test_bm25_grows_with_tf_but_stays_below_ceiling(tf: int, doc_id: int):
    scorer = BM25()
    now = scorer.score("x", Posting(doc_id, tf), index_for_scores)
    more = scorer.score("x", Posting(doc_id, tf + 1), index_for_scores)
    assert more >= now > 0
    # насичення: рейтинг ніколи не доходить до стелі idf * (k1 + 1)
    n, df = index_for_scores.num_docs, index_for_scores.df("x")
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    assert more < idf * (scorer.k1 + 1)


@given(st.integers(1, 500), st.sampled_from([0, 1, 2, 3]))
def test_scorers_never_return_negative_numbers(tf: int, doc_id: int):
    for scorer in (BM25(), TfIdf()):
        for term in ("x", "y", "z", "w", "missing"):
            assert scorer.score(term, Posting(doc_id, tf), index_for_scores) >= 0


@given(
    st.dictionaries(
        st.integers(0, 3), st.floats(0, 100, allow_nan=False, allow_infinity=False), max_size=4
    ),
    st.integers(1, 6),
)
def test_top_k_equals_sorted_prefix(scores: dict[int, float], k: int):
    index = build_index(make_docs(["a", "b", "c", "d"]))
    expected = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    assert [(r.doc_id, r.score) for r in top_k(index, scores, k)] == expected
