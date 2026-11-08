"""precision@5 для TF-IDF і BM25 на 10 запитах із ручною розміткою.

Запуск: python scripts/eval_ranking.py data/index.bin

Розмітка зроблена за змістом книг (про що книга, хто в ній діє): для кожного запиту записано,
які книги корпусу мають бути в топ-5. Запити пишуться через OR (список слів теми), інакше
обидва scorer-и повертали б одну й ту саму малу множину й різниця була б непомітна.
"""

import sys
from pathlib import Path

from findex.measure import setup_console
from findex.ranking import BM25, Scorer, TfIdf
from findex.searcher import Searcher
from findex.store import load

# запит -> id книг Project Gutenberg, які мають бути в топ-5
QUERIES: list[tuple[str, set[str]]] = [
    ("sherlock OR holmes OR watson", {"pg1661", "pg244", "pg2097"}),
    ("zarathustra OR superman OR overman", {"pg1998", "pg7205"}),
    ("valjean OR javert OR cosette OR fantine", {"pg135", "pg17489"}),
    ("hamlet OR ophelia OR polonius OR laertes", {"pg1524", "pg100"}),
    ("whale OR harpoon OR ahab OR pequod", {"pg2701"}),
    ("raskolnikov OR sonia OR porfiry OR razumihin", {"pg2554"}),
    # тематичні запити, розмітка тут "шумніша": це моя думка, які книги про це
    ("marriage OR fortune OR ball OR estate OR gentleman", {"pg1342", "pg158", "pg161"}),
    ("ship OR sailors OR sea OR captain", {"pg2701", "pg120", "pg1727"}),
    ("fairy OR princess OR witch OR enchanted", {"pg2591", "pg55", "pg11"}),
    ("monster OR creature OR vampire OR horror OR laboratory", {"pg84", "pg345", "pg43"}),
]

K = 5


def evaluate(searcher: Searcher, scorer: Scorer, query: str, relevant: set[str]):
    results = searcher.search(query, scorer, K)
    paths = [searcher.index.doc_meta[r.doc_id].path.removesuffix(".txt") for r in results]
    hits = sum(p in relevant for p in paths)
    return paths, hits


def main() -> None:
    setup_console()
    index = load(Path(sys.argv[1]))
    searcher = Searcher(index)
    scorers: dict[str, Scorer] = {"TF-IDF": TfIdf(), "BM25": BM25()}

    print("| # | запит | має бути в топ-5 | стеля P@5 | P@5 TF-IDF | P@5 BM25 |")
    print("|---:|---|---|---:|---:|---:|")
    totals = {name: 0.0 for name in scorers}
    details: list[str] = []
    for i, (query, relevant) in enumerate(QUERIES, start=1):
        row: dict[str, float] = {}
        for name, scorer in scorers.items():
            paths, hits = evaluate(searcher, scorer, query, relevant)
            row[name] = hits / K
            totals[name] += hits / K
            details.append(f"{i:>2} {name:<6} топ-5: {', '.join(paths)}")
        ceiling = min(len(relevant), K) / K
        wanted = ", ".join(sorted(relevant))
        print(
            f"| {i} | `{query}` | {wanted} | {ceiling:.1f} "
            f"| {row['TF-IDF']:.1f} | {row['BM25']:.1f} |"
        )

    n = len(QUERIES)
    ceiling_mean = sum(min(len(r), K) / K for _, r in QUERIES) / n
    print(
        f"| | **середнє** | | {ceiling_mean:.2f} "
        f"| {totals['TF-IDF'] / n:.2f} | {totals['BM25'] / n:.2f} |"
    )
    print("\nфактичний топ-5:")
    print("\n".join(details))


if __name__ == "__main__":
    main()
