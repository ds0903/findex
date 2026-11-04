from pathlib import Path

from findex.documents import Document

TEXTS = [
    "the cat sat on the mat",
    "the dog sat",
    "a cat and a dog",
]


def make_docs(texts=TEXTS):
    # генератор, а не список: індекс має будуватися з потоку
    return (
        Document(doc_id=f"d{i}.txt", path=Path(f"d{i}.txt"), text=text, title=f"Doc {i}")
        for i, text in enumerate(texts)
    )
