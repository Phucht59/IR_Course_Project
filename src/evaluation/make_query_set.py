"""
make_query_set.py – Generate a small, reusable query set for IR evaluation.

Produces evaluation/qrels/query_set.csv with columns: query_id, query_text, aspect.
Includes hand-crafted queries covering common restaurant review aspects plus
optional auto-generated queries sampled from frequent terms in the corpus.

Usage:
    python src/evaluation/make_query_set.py
    python src/evaluation/make_query_set.py --add-corpus-queries --input data/processed/subset_restaurants_20000_clean.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation" / "qrels"

# ── hand-crafted queries ────────────────────────────────────────────
MANUAL_QUERIES = [
    # food (4 queries)
    ("Q001", "best pizza in town", "food"),
    ("Q002", "cold food undercooked steak", "food"),
    ("Q003", "fresh sushi great fish", "food"),
    ("Q004", "bland tasteless food", "food"),
    # service (4 queries)
    ("Q005", "slow service rude waiter", "service"),
    ("Q006", "friendly staff excellent service", "service"),
    ("Q007", "waited too long for food", "service"),
    ("Q008", "attentive waiter great hospitality", "service"),
    # ambience (4 queries)
    ("Q009", "romantic atmosphere date night", "ambience"),
    ("Q010", "noisy crowded restaurant", "ambience"),
    ("Q011", "cozy interior great decor", "ambience"),
    ("Q012", "outdoor patio nice view", "ambience"),
    # price (4 queries)
    ("Q013", "overpriced not worth the money", "price"),
    ("Q014", "great value cheap eats", "price"),
    ("Q015", "expensive but worth every penny", "price"),
    ("Q016", "affordable family friendly budget", "price"),
]


def _corpus_queries(input_path: Path, n: int = 5, start_id: int = 26):
    """Generate extra queries from the most frequent bigrams in the corpus."""
    from sklearn.feature_extraction.text import CountVectorizer

    df = pd.read_parquet(input_path)
    vec = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=200)
    vec.fit(df["text_clean"])
    bigrams = vec.get_feature_names_out()
    # pick evenly spaced bigrams for variety
    step = max(1, len(bigrams) // n)
    extra = []
    for i in range(0, n * step, step):
        qid = f"Q{start_id + len(extra):03d}"
        extra.append((qid, bigrams[i], "auto"))
    return extra[:n]


def make(input_path: Path | None = None, add_corpus: bool = False):
    rows = list(MANUAL_QUERIES)

    if add_corpus and input_path:
        rows.extend(_corpus_queries(input_path))

    df = pd.DataFrame(rows, columns=["query_id", "query_text", "aspect"])

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "query_set.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} queries -> {out_path}")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate query set for IR eval")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Cleaned parquet (needed only with --add-corpus-queries)",
    )
    parser.add_argument(
        "--add-corpus-queries",
        action="store_true",
        help="Add auto-generated queries from frequent corpus bigrams",
    )
    args = parser.parse_args()
    make(input_path=args.input, add_corpus=args.add_corpus_queries)


if __name__ == "__main__":
    main()
