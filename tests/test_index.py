import dataclasses

import pytest

from findex.index import STORAGES, DocMeta, Index, PlainPosting, Posting, build_index
from helpers import make_docs


def as_tuples(index: Index) -> dict:
    return {
        term: [(p.doc_id, p.tf, p.positions) for p in index.iter_postings(term)]
        for term in index.postings
    }


def test_postings_sorted_with_tf(docs):
    index = build_index(docs)
    assert [(p.doc_id, p.tf) for p in index["the"]] == [(0, 2), (1, 1)]
    assert [(p.doc_id, p.tf) for p in index["cat"]] == [(0, 1), (2, 1)]
    assert [(p.doc_id, p.tf) for p in index["a"]] == [(2, 2)]
    for term in index:
        ids = [p.doc_id for p in index[term]]
        assert ids == sorted(ids)


def test_doc_lengths_and_meta(docs):
    index = build_index(docs)
    assert index.doc_lengths == {0: 6, 1: 3, 2: 5}
    assert index.doc_meta[1] == DocMeta(path="d1.txt", title="Doc 1")


def test_missing_term_does_not_create_entry(docs):
    index = build_index(docs)
    assert len(index.doc_ids("zzz")) == 0
    assert index.df("zzz") == 0
    assert "zzz" not in index.postings
    assert type(index.postings) is dict  # не defaultdict


def test_df_and_num_postings(docs):
    index = build_index(docs)
    assert index.df("the") == 2
    assert index.df("mat") == 1
    assert index.num_postings() == sum(index.df(t) for t in index.postings)


@pytest.mark.parametrize("storage", STORAGES)
def test_all_storages_hold_the_same_data(storage):
    reference = as_tuples(build_index(make_docs(), storage="slots"))
    assert as_tuples(build_index(make_docs(), storage=storage)) == reference


def test_storage_types():
    assert isinstance(build_index(make_docs(), "slots")["cat"][0], Posting)
    assert isinstance(build_index(make_docs(), "plain")["cat"][0], PlainPosting)
    entry = build_index(make_docs(), "array").postings["cat"]
    assert isinstance(entry, tuple)  # array-сховище: пара масивів (doc_ids, tfs)
    ids, tfs = entry
    assert list(ids) == [0, 2]
    assert list(tfs) == [1, 1]
    assert ids.typecode == "I"


def test_positions(docs):
    index = build_index(docs, positions=True)
    assert index.positions
    assert [p.positions for p in index["the"]] == [(0, 4), (0,)]
    # tf це кількість позицій
    for term in index:
        assert all(len(p.positions) == p.tf for p in index[term])


def test_without_positions_tuples_are_empty(docs):
    index = build_index(docs)
    assert all(p.positions == () for term in index for p in index[term])


def test_positions_not_supported_for_array(docs):
    with pytest.raises(ValueError):
        build_index(docs, storage="array", positions=True)


def test_unknown_storage(docs):
    with pytest.raises(ValueError):
        build_index(docs, storage="nope")  # type: ignore[arg-type]  # навмисно неправильне значення


def test_empty_corpus():
    index = build_index(iter(()))
    assert index.postings == {}
    assert index.doc_meta == {}
    assert index.all_doc_ids() == range(0)


def test_builds_from_a_stream_one_document_at_a_time():
    seen = []

    def stream():
        for doc in make_docs():
            seen.append(doc.doc_id)
            yield doc

    gen = stream()
    assert seen == []  # ще нічого не прочитано
    index = build_index(gen)
    assert seen == ["d0.txt", "d1.txt", "d2.txt"]
    assert index.df("sat") == 2


# --- контракт hash/eq і slots (те, що вимагає лаба) ---


def test_posting_and_docmeta_are_hashable_and_usable_in_sets():
    assert len({Posting(1, 2), Posting(1, 2), Posting(1, 3)}) == 2
    assert len({DocMeta("a", "A"), DocMeta("a", "A")}) == 1
    assert {Posting(1, 2, (0, 5)): "x"}[Posting(1, 2, (0, 5))] == "x"


def test_equal_objects_have_equal_hashes():
    a, b = Posting(7, 3, (1, 2, 3)), Posting(7, 3, (1, 2, 3))
    assert a == b
    assert hash(a) == hash(b)


def test_frozen():
    posting = Posting(1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        posting.tf = 5  # type: ignore[misc]  # навмисна спроба змінити frozen-об'єкт
    with pytest.raises(dataclasses.FrozenInstanceError):
        DocMeta("a", "b").title = "c"  # type: ignore[misc]  # те саме


def test_slots_instances_have_no_dict():
    assert not hasattr(Posting(1, 2), "__dict__")
    assert not hasattr(DocMeta("a", "b"), "__dict__")
    assert hasattr(PlainPosting(1, 2), "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        Posting(1, 2).extra = 1  # type: ignore[attr-defined]  # slots не дозволяють нових полів
