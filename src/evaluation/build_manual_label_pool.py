"""
build_manual_label_pool.py – Create a CSV file for manual relevance labeling.

Pools top-k documents from TF-IDF and BM25 run files for each query,
joins review text and business metadata from the cleaned parquet, and
outputs a CSV ready for human annotation.

Usage:
    python src/evaluation/build_manual_label_pool.py

    python src/evaluation/build_manual_label_pool.py ^
        --query-set  evaluation/qrels/query_set.csv ^
        --run-tfidf  evaluation/runs/run_tfidf.txt ^
        --run-bm25   evaluation/runs/run_bm25.txt ^
        --data       data/processed/subset_restaurants_20000_clean.parquet ^
        --top-k 10 ^
        --output     evaluation/qrels/manual_label_pool.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# ── helpers ─────────────────────────────────────────────────────────
def load_run_topk(path: Path, k: int) -> dict[str, list[str]]:
    """Return {query_id: [doc_id, …]} keeping only top-k per query."""
    run: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, did, rank = parts[0], parts[2], int(parts[3])
            run[qid].append((rank, did))
    return {qid: [did for _, did in sorted(docs)[:k]] for qid, docs in run.items()}


# ── main logic ──────────────────────────────────────────────────────
def build_pool(
    query_set_path: Path,
    run_tfidf_path: Path,
    run_bm25_path: Path,
    data_path: Path,
    top_k: int,
    output_path: Path,
):
    # 1. Load query set
    queries = pd.read_csv(query_set_path)
    print(f"Loaded {len(queries)} queries from {query_set_path}")

    # 2. Load run files (top-k per query)
    tfidf_run = load_run_topk(run_tfidf_path, top_k)
    bm25_run = load_run_topk(run_bm25_path, top_k)
    print(f"Loaded runs  tfidf={len(tfidf_run)} queries, bm25={len(bm25_run)} queries")

    # 3. Pool documents per query, tracking source systems
    pool_rows: list[dict] = []
    for _, qrow in queries.iterrows():
        qid = qrow["query_id"]
        qtxt = qrow["query_text"]

        tfidf_docs = set(tfidf_run.get(qid, []))
        bm25_docs = set(bm25_run.get(qid, []))
        all_docs = tfidf_docs | bm25_docs

        for doc_id in sorted(all_docs):
            sources = []
            if doc_id in tfidf_docs:
                sources.append("tfidf")
            if doc_id in bm25_docs:
                sources.append("bm25")

            pool_rows.append(
                {
                    "query_id": qid,
                    "query_text": qtxt,
                    "doc_id": doc_id,
                    "source_systems": "+".join(sources),
                }
            )

    pool = pd.DataFrame(pool_rows)
    print(
        f"Pooled {len(pool)} (query, doc) pairs  ({pool['doc_id'].nunique()} unique docs)"
    )

    # 4. Join review metadata from the parquet
    data = pd.read_parquet(
        data_path, columns=["review_id", "name", "stars", "sentiment", "text"]
    )
    data = data.rename(
        columns={"review_id": "doc_id", "name": "business_name", "text": "review_text"}
    )

    pool = pool.merge(data, on="doc_id", how="left")

    # 5. Add blank annotation columns
    pool["label"] = ""
    pool["notes"] = ""

    # 6. Reorder columns
    col_order = [
        "query_id",
        "query_text",
        "doc_id",
        "business_name",
        "stars",
        "sentiment",
        "review_text",
        "source_systems",
        "label",
        "notes",
    ]
    pool = pool[col_order]

    # 7. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(output_path, index=False)
    print(f"\nSaved manual label pool -> {output_path}")
    print(f"  Queries : {pool['query_id'].nunique()}")
    print(f"  Pairs   : {len(pool)}")
    print(
        f"\nNext step: open the CSV, fill in the 'label' column (0 = not relevant, 1 = relevant, 2 = highly relevant),"
    )
    print(f"  then convert to TREC qrels format with:")
    print(f"    query_id  0  doc_id  label")


def main():
    parser = argparse.ArgumentParser(
        description="Build a document pool for manual relevance labeling",
    )
    parser.add_argument(
        "--query-set",
        type=Path,
        default=ROOT / "evaluation" / "qrels" / "query_set.csv",
        help="CSV with columns query_id, query_text, aspect",
    )
    parser.add_argument(
        "--run-tfidf",
        type=Path,
        default=ROOT / "evaluation" / "runs" / "run_tfidf.txt",
        help="TREC-style run file from TF-IDF retrieval",
    )
    parser.add_argument(
        "--run-bm25",
        type=Path,
        default=ROOT / "evaluation" / "runs" / "run_bm25.txt",
        help="TREC-style run file from BM25 retrieval",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet",
        help="Cleaned review parquet with review_id, name, stars, sentiment, text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top documents to pool per system per query (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "qrels" / "manual_label_pool.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    build_pool(
        query_set_path=args.query_set,
        run_tfidf_path=args.run_tfidf,
        run_bm25_path=args.run_bm25,
        data_path=args.data,
        top_k=args.top_k,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
