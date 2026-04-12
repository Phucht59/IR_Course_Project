"""
build_bm25.py – Build and persist a BM25 index using rank_bm25.

Usage:
    python src/indexing/build_bm25.py --input data/processed/subset_restaurants_20000_clean.parquet
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from rank_bm25 import BM25Okapi

MODELS = Path(__file__).resolve().parents[2] / "models"


def build(input_path: Path):
    print(f"Reading {input_path} …")
    df = pd.read_parquet(input_path)

    print("Tokenising …")
    corpus = df["text_clean"].tolist()
    tokenized = [doc.split() for doc in corpus]

    print("Building BM25 index …")
    bm25 = BM25Okapi(tokenized)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bm25, MODELS / "bm25_index.pkl")

    meta = df[["review_id", "business_id", "stars", "sentiment", "name", "text"]].copy()
    meta.to_parquet(MODELS / "bm25_meta.parquet", index=False)

    print(f"Done – {len(corpus):,} docs indexed, saved to {MODELS}")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index")
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    build(args.input)


if __name__ == "__main__":
    main()
