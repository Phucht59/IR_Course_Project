"""
make_pseudo_qrels.py – Build pseudo-relevance judgments and TREC-style run files
using keyword-overlap heuristics.

For each query in the query set:
  1. Run both TF-IDF and BM25 retrieval.
  2. Pool top-N candidates (union-deduplicated).
  3. Score each candidate with keyword-overlap heuristics -> graded relevance (0/1/2).
  4. Write TREC-style qrels and run files.

Usage:
    python src/evaluation/make_pseudo_qrels.py \
        --input data/processed/subset_restaurants_20000_clean.parquet \
        --queries evaluation/qrels/query_set.csv
"""

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
QRELS = ROOT / "evaluation" / "qrels"
RUNS = ROOT / "evaluation" / "runs"


# ── helpers ─────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


SYNONYMS: dict[str, list[str]] = {
    "slow": ["slow", "waited", "forever", "long wait", "took forever", "delayed"],
    "rude": ["rude", "unfriendly", "disrespectful", "nasty", "attitude"],
    "friendly": ["friendly", "nice", "kind", "welcoming", "pleasant", "helpful"],
    "dirty": ["dirty", "filthy", "disgusting", "gross", "unclean", "unsanitary"],
    "overpriced": ["overpriced", "expensive", "ripoff", "rip off", "not worth"],
    "cheap": ["cheap", "affordable", "bargain", "inexpensive", "value"],
    "cold": ["cold", "lukewarm", "not hot", "room temperature"],
    "fresh": ["fresh", "crispy", "tender", "juicy"],
    "bland": ["bland", "tasteless", "no flavor", "no taste", "flavorless"],
    "amazing": ["amazing", "incredible", "outstanding", "fantastic", "phenomenal"],
    "worst": ["worst", "terrible", "horrible", "awful", "nightmare"],
    "best": ["best", "greatest", "favorite", "favourite", "top notch", "excellent"],
    "noisy": ["noisy", "loud", "crowded", "packed"],
    "cozy": ["cozy", "intimate", "comfortable", "warm", "charming"],
    "vegan": ["vegan", "vegetarian", "plant based", "meatless"],
    "gluten": ["gluten free", "celiac", "gluten-free"],
}


def _expand_query_keywords(query: str) -> set[str]:
    """Return a broad set of matching keywords including synonyms."""
    tokens = _clean(query).split()
    expanded = set(tokens)
    for tok in tokens:
        if tok in SYNONYMS:
            expanded.update(SYNONYMS[tok])
    return expanded


def _keyword_relevance(doc_text: str, keywords: set[str]) -> int:
    """
    Graded relevance:
      2 = doc contains ≥60 % of expanded keywords  (highly relevant)
      1 = doc contains ≥30 % of expanded keywords  (partially relevant)
      0 = below threshold
    """
    doc_lower = doc_text.lower()
    hits = sum(1 for kw in keywords if kw in doc_lower)
    ratio = hits / max(len(keywords), 1)
    if ratio >= 0.60:
        return 2
    if ratio >= 0.30:
        return 1
    return 0


# ── retrieval helpers ───────────────────────────────────────────────
def _retrieve_tfidf(query: str, vec, matrix, top_n: int) -> list[tuple[int, float]]:
    q = vec.transform([_clean(query)])
    scores = cosine_similarity(q, matrix).flatten()
    idx = np.argsort(scores)[::-1][:top_n]
    return [(int(i), float(scores[i])) for i in idx]


def _retrieve_bm25(query: str, bm25, top_n: int) -> list[tuple[int, float]]:
    tokens = _clean(query).split()
    scores = bm25.get_scores(tokens)
    idx = np.argsort(scores)[::-1][:top_n]
    return [(int(i), float(scores[i])) for i in idx]


# ── main pipeline ──────────────────────────────────────────────────
def make(
    input_path: Path,
    queries_path: Path,
    pool_depth: int = 20,
):
    # load corpus
    df = pd.read_parquet(input_path)

    # load indexes
    vec = joblib.load(MODELS / "tfidf_vectorizer.pkl")
    tfidf_matrix = sparse.load_npz(MODELS / "tfidf_matrix.npz")
    bm25 = joblib.load(MODELS / "bm25_index.pkl")

    # load query set
    queries = pd.read_csv(queries_path)
    print(f"Loaded {len(queries)} queries, {len(df):,} docs")

    qrels_lines: list[str] = []
    run_tfidf_lines: list[str] = []
    run_bm25_lines: list[str] = []

    for _, qrow in queries.iterrows():
        qid = qrow["query_id"]
        qtext = qrow["query_text"]
        keywords = _expand_query_keywords(qtext)

        # retrieve candidates from both systems
        tfidf_hits = _retrieve_tfidf(qtext, vec, tfidf_matrix, pool_depth)
        bm25_hits = _retrieve_bm25(qtext, bm25, pool_depth)

        # write run files
        for rank, (idx, score) in enumerate(tfidf_hits, 1):
            rid = df.iloc[idx]["review_id"]
            run_tfidf_lines.append(f"{qid} Q0 {rid} {rank} {score:.4f} tfidf_baseline")

        for rank, (idx, score) in enumerate(bm25_hits, 1):
            rid = df.iloc[idx]["review_id"]
            run_bm25_lines.append(f"{qid} Q0 {rid} {rank} {score:.4f} bm25_baseline")

        # pool & judge
        pooled_indices = dict(tfidf_hits)
        pooled_indices.update(dict(bm25_hits))  # union

        for idx in pooled_indices:
            doc_text = df.iloc[idx]["text_clean"]
            rid = df.iloc[idx]["review_id"]
            rel = _keyword_relevance(doc_text, keywords)
            qrels_lines.append(f"{qid} 0 {rid} {rel}")

    # save
    QRELS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    qrels_path = QRELS / "qrels_pseudo.txt"
    qrels_path.write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")

    run_tfidf_path = RUNS / "run_tfidf.txt"
    run_tfidf_path.write_text("\n".join(run_tfidf_lines) + "\n", encoding="utf-8")

    run_bm25_path = RUNS / "run_bm25.txt"
    run_bm25_path.write_text("\n".join(run_bm25_lines) + "\n", encoding="utf-8")

    n_rel = sum(1 for l in qrels_lines if l.strip().split()[-1] != "0")
    print(
        f"\nPseudo-qrels : {len(qrels_lines)} judgments ({n_rel} relevant) -> {qrels_path}"
    )
    print(f"Run TF-IDF   : {len(run_tfidf_lines)} rows -> {run_tfidf_path}")
    print(f"Run BM25     : {len(run_bm25_lines)} rows -> {run_bm25_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo qrels + run files")
    parser.add_argument(
        "--input", required=True, type=Path, help="Cleaned parquet file"
    )
    parser.add_argument(
        "--queries", type=Path, default=ROOT / "evaluation" / "qrels" / "query_set.csv"
    )
    parser.add_argument(
        "--pool-depth", type=int, default=20, help="Top-N per system for pooling"
    )
    args = parser.parse_args()
    make(args.input, args.queries, pool_depth=args.pool_depth)


if __name__ == "__main__":
    main()
