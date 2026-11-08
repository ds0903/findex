"""Збереження й завантаження індексу у двох форматах.

pickle - просто і швидко пишеться, але небезпечно читати чужі файли.
binary - свій формат: JSON-заголовок + масиви array('I') як є, без виконання коду.

Формат binary:
    MAGIC (8 байтів) | довжина заголовка, uint32 LE | заголовок (JSON, utf-8) | масиви
    масиви (по порядку): counts (скільки постінгів у кожного терма), doc_ids, tfs,
    а якщо є позиції - ще positions (для кожного постінга tf чисел).
"""

import json
import pickle
import struct
import sys
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from findex.index import STORAGES, DocMeta, Entry, Index, PlainPosting, Posting, Storage
from findex.timing import timed

type Format = Literal["pickle", "binary"]
FORMATS: tuple[Format, ...] = ("pickle", "binary")
MAGIC = b"FINDEX02"


def save(index: Index, path: Path | str, fmt: Format = "pickle") -> None:
    if fmt == "pickle":
        with open(path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "binary":
        _save_binary(index, Path(path))
    else:
        raise ValueError(f"невідомий формат {fmt!r}, є: {', '.join(FORMATS)}")


@timed
def load(path: Path | str) -> Index:
    """Читає індекс, формат визначає за вмістом файлу (а не за розширенням).

    УВАГА: pickle.load не можна викликати для файлів, яким ви не довіряєте.
    Pickle - це не дані, а програма для віртуальної машини pickle: файл може
    містити інструкцію "викличи os.system(...)" (через __reduce__), і вона
    виконається в момент load, ще до того як ми щось перевірили. Тому pickle
    годиться тільки для власних файлів. Формат binary нижче такого не вміє:
    він читає лише JSON і сирі масиви чисел.
    """
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(len(MAGIC))
        if head == MAGIC:
            return _load_binary(path)
        f.seek(0)
        try:
            obj: object = pickle.load(f)  # саме тут виконався б чужий код, див. вище
        except Exception as exc:
            raise ValueError(f"{path}: це не файл індексу findex") from exc
    if not isinstance(obj, Index):
        raise ValueError(f"{path}: в pickle лежить не Index, а {type(obj).__name__}")
    return obj


@contextmanager
def open_index(path: Path | str) -> Iterator[Index]:
    """with open_index(path) as index: ...

    Після виходу з with (і нормального, і через виняток) індекс закривається:
    його словники очищаються, пам'ять звільняється.
    """
    index = load(path)
    try:
        yield index
    finally:
        index.close()


def _save_binary(index: Index, path: Path) -> None:
    terms = sorted(index.postings)
    counts: array[int] = array("I")
    doc_ids: array[int] = array("I")
    tfs: array[int] = array("I")
    positions: array[int] = array("I")

    for term in terms:
        entry = index.postings[term]
        if isinstance(entry, tuple):
            counts.append(len(entry[0]))
            doc_ids.extend(entry[0])
            tfs.extend(entry[1])
        else:
            counts.append(len(entry))
            for p in entry:
                doc_ids.append(p.doc_id)
                tfs.append(p.tf)
                if index.positions:
                    positions.extend(p.positions)

    header = {
        "storage": index.storage,
        "positions": index.positions,
        "root": index.root,
        "byteorder": sys.byteorder,
        "itemsize": doc_ids.itemsize,
        "terms": terms,
        "docs": [
            [doc_id, meta.path, meta.title, index.doc_lengths[doc_id]]
            for doc_id, meta in sorted(index.doc_meta.items())
        ],
    }
    blob = json.dumps(header, ensure_ascii=False).encode("utf-8")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(blob)))
        f.write(blob)
        for arr in (counts, doc_ids, tfs, positions):
            f.write(arr.tobytes())


@dataclass(frozen=True)
class _Header:
    storage: Storage
    positions: bool
    root: str
    byteorder: str
    itemsize: int
    terms: list[str]
    docs: list[tuple[int, str, str, int]]


def _parse_header(raw: bytes, path: Path) -> _Header:
    try:
        data = cast("dict[str, Any]", json.loads(raw))
        storage = str(data["storage"])
        if storage not in STORAGES:
            raise ValueError(storage)
        return _Header(
            storage=cast("Storage", storage),
            positions=bool(data["positions"]),
            root=str(data.get("root", "")),  # у файлах лаби 2 цього ключа нема
            byteorder=str(data["byteorder"]),
            itemsize=int(data["itemsize"]),
            terms=[str(t) for t in data["terms"]],
            docs=[(int(d), str(p), str(t), int(n)) for d, p, t, n in data["docs"]],
        )
    except (KeyError, ValueError, TypeError) as exc:  # JSONDecodeError теж ValueError
        raise ValueError(f"{path}: пошкоджений заголовок") from exc


def _load_binary(path: Path) -> Index:
    data = memoryview(path.read_bytes())
    start = len(MAGIC)
    try:
        (header_len,) = struct.unpack_from("<I", data, start)
    except struct.error as exc:
        raise ValueError(f"{path}: пошкоджений заголовок") from exc
    start += 4
    header = _parse_header(bytes(data[start : start + header_len]), path)
    if header.itemsize != array("I").itemsize:
        raise ValueError(f"{path}: розмір unsigned int на цій машині інший")

    offset = start + header_len

    def take(n: int) -> array[int]:
        nonlocal offset
        arr: array[int] = array("I")
        chunk = data[offset : offset + n * arr.itemsize]
        if len(chunk) != n * arr.itemsize:
            raise ValueError(f"{path}: файл обрізаний")
        arr.frombytes(chunk)
        if header.byteorder != sys.byteorder:
            arr.byteswap()
        offset += n * arr.itemsize
        return arr

    counts = take(len(header.terms))
    total = sum(counts)
    doc_ids = take(total)
    tfs = take(total)
    positions = take(sum(tfs)) if header.positions else array("I")

    postings: dict[str, Entry] = {}
    i = 0  # де ми в doc_ids/tfs
    p = 0  # де ми в positions
    for term, n in zip(header.terms, counts, strict=True):
        if header.storage == "array":
            postings[term] = (doc_ids[i : i + n], tfs[i : i + n])
        else:
            rows: list[tuple[int, int, tuple[int, ...]]] = []
            for k in range(i, i + n):
                tf = tfs[k]
                where: tuple[int, ...] = ()
                if header.positions:
                    where = tuple(positions[p : p + tf])
                    p += tf
                rows.append((doc_ids[k], tf, where))
            if header.storage == "plain":
                postings[term] = [PlainPosting(*row) for row in rows]
            else:
                postings[term] = [Posting(*row) for row in rows]
        i += n

    doc_meta = {doc_id: DocMeta(doc_path, title) for doc_id, doc_path, title, _ in header.docs}
    doc_lengths = {doc_id: length for doc_id, _, _, length in header.docs}
    return Index(postings, doc_lengths, doc_meta, header.storage, header.positions, header.root)
