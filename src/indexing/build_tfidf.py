"""
build_tfidf.py – Fit a TF-IDF vectorizer and persist the sparse matrix.

Usage:
    python src/indexing/build_tfidf.py --input data/processed/subset_restaurants_20000_clean.parquet
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS = Path(__file__).resolve().parents[2] / "models"


def build(input_path: Path, max_features: int = 50_000):
    print(f"Reading {input_path} …")
    df = pd.read_parquet(input_path)

    print("Fitting TF-IDF …")
    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = vec.fit_transform(df["text_clean"])

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, MODELS / "tfidf_vectorizer.pkl")
    sparse.save_npz(MODELS / "tfidf_matrix.npz", tfidf_matrix)

    # save lightweight metadata for retrieval lookup
    meta = df[["review_id", "business_id", "stars", "sentiment", "name", "text"]].copy()
    meta.to_parquet(MODELS / "tfidf_meta.parquet", index=False)

    print(f"Done – matrix shape {tfidf_matrix.shape}, saved to {MODELS}")


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF index")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max-features", type=int, default=50_000)
    args = parser.parse_args()
    build(args.input, max_features=args.max_features)


if __name__ == "__main__":
    main()
