"""Три перевірки рейтингу на справжньому корпусі.

Запуск: python scripts/sanity_ranking.py data/index.bin
"""

import sys
from pathlib import Path

from findex.index import Index, Posting, PostingLike
from findex.measure import setup_console
from findex.ranking import BM25, Scorer, TfIdf
from findex.searcher import Searcher
from findex.store import load


def title(index: Index, doc_id: int) -> str:
    return index.doc_meta[doc_id].title


def posting_of(index: Index, term: str, doc_id: int) -> PostingLike:
    return next(p for p in index.iter_postings(term) if p.doc_id == doc_id)


def tf_of(index: Index, term: str, doc_id: int) -> int:
    """Скільки разів терм є в документі (0, якщо нема)."""
    return next((p.tf for p in index.iter_postings(term) if p.doc_id == doc_id), 0)


def check1(index: Index, scorers: dict[str, Scorer]) -> None:
    print("## 1. рідкий терм вище за частий\n")
    rare, common = "harpoon", "in"
    print(
        f"`{rare}`: df = {index.df(rare)} з {index.num_docs}; `{common}`: df = {index.df(common)}\n"
    )
    searcher = Searcher(index)
    query = f"{rare} OR {common}"
    with_rare = {p.doc_id for p in index.iter_postings(rare)}
    for name, scorer in scorers.items():
        results = searcher.search(query, scorer, k=index.num_docs)
        top = results[0]
        best_without = next(r for r in results if r.doc_id not in with_rare)
        p_rare = posting_of(index, rare, top.doc_id)
        p_common = posting_of(index, common, top.doc_id)
        print(f"{name}, запит `{query}`:")
        print(f"  1-ше місце: {top.title}, {top.score:.3f} (є `{rare}`)")
        print(f"  найкращий документ без `{rare}`: {best_without.title}, {best_without.score:.3f}")
        print(
            f"  у документі-лідері: внесок `{rare}` = {scorer.score(rare, p_rare, index):.3f} "
            f"(tf = {p_rare.tf}), внесок `{common}` = {scorer.score(common, p_common, index):.3f} "
            f"(tf = {p_common.tf})"
        )
    print()


def check2(index: Index, scorers: dict[str, Scorer]) -> None:
    print("## 2. 20-те повторення майже нічого не додає\n")
    term = "whale"
    doc_id = next(i for i, m in index.doc_meta.items() if m.path == "pg2701.txt")
    print(
        f"`{term}` (df = {index.df(term)}), документ «{title(index, doc_id)}», "
        f"довжина {index.doc_length(doc_id)} токенів, середня {index.avg_doc_length:.0f}. "
        "tf підставляємо штучно, idf і довжина документа справжні.\n"
    )

    def score(name: str, tf: int) -> float:
        return scorers[name].score(term, Posting(doc_id, tf), index) if tf else 0.0

    print("| tf | BM25 | що додало це повторення | TF-IDF | що додало це повторення |")
    print("|---:|---:|---:|---:|---:|")
    for tf in (1, 2, 3, 5, 10, 20, 21, 50, 100):
        bm, ti = score("BM25", tf), score("TF-IDF", tf)
        bm_gain, ti_gain = bm - score("BM25", tf - 1), ti - score("TF-IDF", tf - 1)
        print(f"| {tf} | {bm:.3f} | +{bm_gain:.4f} | {ti:.3f} | +{ti_gain:.4f} |")
    print()


def check3(index: Index, scorers: dict[str, Scorer]) -> None:
    print("## 3. короткий документ з одним входженням вище за дуже довгий\n")
    lengths = index.doc_lengths
    short_id = min(lengths, key=lambda d: lengths[d])
    long_id = max(lengths, key=lambda d: lengths[d])
    print(
        f"найкоротший: «{title(index, short_id)}», {lengths[short_id]} токенів; "
        f"найдовший: «{title(index, long_id)}», {lengths[long_id]} токенів\n"
    )
    shared = [t for t in index if tf_of(index, t, short_id) == 1 and tf_of(index, t, long_id) == 1]
    term = min(shared, key=lambda t: (index.df(t), t))
    print(f"терм, що є в обох рівно по одному разу: `{term}` (df = {index.df(term)})\n")
    print("| scorer | короткий | довгий |")
    print("|---|---:|---:|")
    for name, scorer in scorers.items():
        short = scorer.score(term, posting_of(index, term, short_id), index)
        long_ = scorer.score(term, posting_of(index, term, long_id), index)
        print(f"| {name} | {short:.3f} | {long_:.3f} |")
    print()


def main() -> None:
    setup_console()
    index = load(Path(sys.argv[1]))
    scorers: dict[str, Scorer] = {"BM25": BM25(), "TF-IDF": TfIdf()}
    check1(index, scorers)
    check2(index, scorers)
    check3(index, scorers)


if __name__ == "__main__":
    main()
