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
from pathlib import Path

from findex.index import DocMeta, Index, PlainPosting, Posting

FORMATS = ("pickle", "binary")
MAGIC = b"FINDEX02"


def save(index: Index, path: Path | str, fmt: str = "pickle") -> None:
    if fmt == "pickle":
        with open(path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "binary":
        _save_binary(index, Path(path))
    else:
        raise ValueError(f"невідомий формат {fmt!r}, є: {', '.join(FORMATS)}")


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
            index = pickle.load(f)  # саме тут виконався б чужий код, див. попередження вище
        except Exception as exc:
            raise ValueError(f"{path}: це не файл індексу findex") from exc
    if not isinstance(index, Index):
        raise ValueError(f"{path}: в pickle лежить не Index, а {type(index).__name__}")
    return index


def _save_binary(index: Index, path: Path) -> None:
    terms = sorted(index.postings)
    counts = array("I")
    doc_ids = array("I")
    tfs = array("I")
    positions = array("I")

    for term in terms:
        entry = index.postings[term]
        if index.storage == "array":
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


def _load_binary(path: Path) -> Index:
    data = memoryview(path.read_bytes())
    start = len(MAGIC)
    try:
        (header_len,) = struct.unpack_from("<I", data, start)
        start += 4
        header = json.loads(bytes(data[start : start + header_len]))
        terms = header["terms"]
        storage = header["storage"]
        has_positions = header["positions"]
        docs = header["docs"]
        byteorder = header["byteorder"]
        if header["itemsize"] != array("I").itemsize:
            raise ValueError("розмір unsigned int на цій машині інший")
    except (struct.error, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: пошкоджений заголовок") from exc

    offset = start + header_len

    def take(n: int) -> array:
        nonlocal offset
        arr = array("I")
        chunk = data[offset : offset + n * arr.itemsize]
        if len(chunk) != n * arr.itemsize:
            raise ValueError(f"{path}: файл обрізаний")
        arr.frombytes(chunk)
        if byteorder != sys.byteorder:
            arr.byteswap()
        offset += n * arr.itemsize
        return arr

    counts = take(len(terms))
    total = sum(counts)
    doc_ids = take(total)
    tfs = take(total)
    positions = take(sum(tfs)) if has_positions else array("I")

    posting_cls = PlainPosting if storage == "plain" else Posting
    postings: dict = {}
    i = 0  # де ми в doc_ids/tfs
    p = 0  # де ми в positions
    for term, n in zip(terms, counts, strict=True):
        if storage == "array":
            postings[term] = (doc_ids[i : i + n], tfs[i : i + n])
        else:
            items = []
            for k in range(i, i + n):
                tf = tfs[k]
                where = ()
                if has_positions:
                    where = tuple(positions[p : p + tf])
                    p += tf
                items.append(posting_cls(doc_ids[k], tf, where))
            postings[term] = items
        i += n

    doc_meta = {doc_id: DocMeta(doc_path, title) for doc_id, doc_path, title, _ in docs}
    doc_lengths = {doc_id: length for doc_id, _, _, length in docs}
    return Index(postings, doc_lengths, doc_meta, storage, has_positions)
