import pickle
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.documents import find_paths, iter_documents
from findex.index import STORAGES, Index, Storage, build_index
from findex.parallel import (
    EXECUTOR_NAMES,
    ExecutorName,
    build_parallel,
    build_partial,
    merge,
    plan_chunks,
)

TEXTS = [
    "the cat sat on the mat",
    "the dog sat",
    "a cat and a dog",
    "straße naïve кіт the end",
    "one more text with the cat",
    "last",
]


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "books"
    root.mkdir()
    for i, text in enumerate(TEXTS):
        (root / f"b{i}.txt").write_text(f"Title: Book {i}\n\n{text}\n", encoding="utf-8")
    return root


def reference(root: Path, storage: Storage = "slots", positions: bool = False) -> Index:
    return build_index(iter_documents(root), storage, positions, root=str(root))


# --- шматки ---


def test_plan_chunks_are_contiguous_and_cover_everything():
    paths = [f"f{i}.txt" for i in range(10)]
    chunks = plan_chunks(paths, 4, [100] * 10)
    assert [p for c in chunks for p in c.paths] == paths
    assert [c.first_doc_id for c in chunks] == [
        sum(len(x.paths) for x in chunks[:i]) for i in range(len(chunks))
    ]
    assert 1 <= len(chunks) <= 4


def test_plan_chunks_balances_by_size_not_by_count():
    paths = ["big.txt", "a.txt", "b.txt", "c.txt"]
    chunks = plan_chunks(paths, 2, [1000, 10, 10, 10])
    assert chunks[0].paths == ("big.txt",)
    assert chunks[1].paths == ("a.txt", "b.txt", "c.txt")


def test_plan_chunks_never_more_chunks_than_files():
    chunks = plan_chunks(["a", "b"], 10, [1, 1])
    assert [c.paths for c in chunks] == [("a",), ("b",)]
    assert plan_chunks([], 3, []) == []


def test_plan_chunks_rejects_bad_input():
    with pytest.raises(ValueError):
        plan_chunks(["a"], 0, [1])
    with pytest.raises(ValueError):
        plan_chunks(["a", "b"], 1, [1])


# --- build_partial і merge ---


def test_build_partial_is_picklable_and_uses_first_doc_id(folder: Path):
    paths = [str(p) for p in find_paths(folder)]
    assert pickle.loads(pickle.dumps(build_partial)) is build_partial  # доступна за ім'ям
    part = build_partial(str(folder), paths[2:4], 2)
    assert sorted(part.doc_meta) == [2, 3]
    assert part.doc_meta[2].path == "b2.txt"
    assert pickle.loads(pickle.dumps(part)) == part  # результат теж повертається з воркера


@pytest.mark.parametrize("storage", STORAGES)
def test_serial_merge_of_one_partial_equals_plain_build(folder: Path, storage: Storage):
    paths = [str(p) for p in find_paths(folder)]
    assert merge([build_partial(str(folder), paths, 0, storage)]) == reference(folder, storage)


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("positions", [False, True])
def test_merge_of_pieces_equals_plain_build(folder: Path, storage: Storage, positions: bool):
    if storage == "array" and positions:
        pytest.skip("array не підтримує позиції")
    paths = [str(p) for p in find_paths(folder)]
    parts = [
        build_partial(str(folder), paths[:2], 0, storage, positions),
        build_partial(str(folder), paths[2:5], 2, storage, positions),
        build_partial(str(folder), paths[5:], 5, storage, positions),
    ]
    assert merge(parts) == reference(folder, storage, positions)


def test_merge_keeps_postings_sorted(folder: Path):
    paths = [str(p) for p in find_paths(folder)]
    merged = merge(
        [build_partial(str(folder), paths[:3], 0), build_partial(str(folder), paths[3:], 3)]
    )
    for term in merged:
        ids = [p.doc_id for p in merged[term]]
        assert ids == sorted(ids)


def test_merge_rejects_overlapping_ranges(folder: Path):
    paths = [str(p) for p in find_paths(folder)]
    with pytest.raises(ValueError, match="перетинаються"):
        merge([build_partial(str(folder), paths[:2], 0), build_partial(str(folder), paths[2:4], 1)])


def test_merge_rejects_mixed_parameters_and_empty(folder: Path):
    paths = [str(p) for p in find_paths(folder)]
    with pytest.raises(ValueError):
        merge([])
    with pytest.raises(ValueError, match="різними параметрами"):
        merge(
            [
                build_partial(str(folder), paths[:2], 0, "slots"),
                build_partial(str(folder), paths[2:], 2, "plain"),
            ]
        )


@given(
    st.lists(st.sampled_from(["cat", "dog", "sat", "мат", "the", "a"]), min_size=1, max_size=8),
    st.lists(st.integers(1, 5), min_size=1, max_size=4),
)
def test_any_split_merges_to_the_same_index(words: list[str], sizes: list[int]):
    from findex.documents import Document

    texts = [" ".join(words[: i + 1]) for i in range(sum(sizes))]
    docs = [Document(f"d{i}", Path(f"d{i}"), t, f"T{i}") for i, t in enumerate(texts)]
    parts: list[Index] = []
    start = 0
    for size in sizes:
        parts.append(build_index(docs[start : start + size], first_doc_id=start))
        start += size
    assert merge(parts) == build_index(docs)


# --- executor-и ---


@pytest.mark.parametrize("executor", EXECUTOR_NAMES)
def test_every_executor_gives_the_same_index(folder: Path, executor: ExecutorName):
    result = build_parallel(folder, find_paths(folder), workers=2, executor=executor)
    assert result.index == reference(folder)
    assert result.merge_seconds >= 0 and result.build_seconds > 0


def test_worker_cpu_time_is_reported_for_processes_only(folder: Path):
    # на крихітному корпусі таймер CPU (крок ≈15 мс на windows) може показати й 0
    result = build_parallel(folder, find_paths(folder), workers=2, executor="processes")
    assert result.worker_cpu_seconds >= 0
    assert build_parallel(folder, find_paths(folder), 2, "threads").worker_cpu_seconds == 0
    assert build_parallel(folder, find_paths(folder), 2, "serial").worker_cpu_seconds == 0


def test_progress_callback_counts_every_file(folder: Path):
    done: list[int] = []
    build_parallel(folder, find_paths(folder), 2, "threads", on_chunk_done=done.append)
    assert sum(done) == len(TEXTS)


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_worker_failure_fails_the_build_with_the_cause(folder: Path, executor: ExecutorName):
    paths = find_paths(folder)
    (folder / "b3.txt").write_bytes(b"\xff\xfe not utf-8 \x80\x81")  # не декодується
    with pytest.raises(ValueError, match="b3.txt"):
        build_parallel(folder, paths, workers=2, executor=executor)


def test_vanished_file_fails_before_any_worker_starts(folder: Path):
    paths = find_paths(folder)
    paths[2].unlink()
    with pytest.raises(FileNotFoundError, match="b2.txt"):  # не вдалося дізнатися розмір
        build_parallel(folder, paths, workers=2, executor="processes")


def test_build_parallel_validates_arguments(folder: Path):
    paths = find_paths(folder)
    with pytest.raises(ValueError):
        build_parallel(folder, paths, workers=0)
    with pytest.raises(ValueError):
        build_parallel(folder, [], workers=1)
    with pytest.raises(ValueError):
        build_parallel(folder, paths, executor="magic")  # type: ignore[arg-type]  # навмисно
