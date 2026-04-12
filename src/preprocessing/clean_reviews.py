"""
clean_reviews.py – Light text cleaning on the subset.

Usage:
    python src/preprocessing/clean_reviews.py --input data/processed/subset_restaurants_20000.parquet
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    """Lowercase, strip URLs, remove punctuation, normalise whitespace."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset(input_path: Path) -> pd.DataFrame:
    print(f"Reading {input_path} …")
    df = pd.read_parquet(input_path)

    print("Cleaning text …")
    df["text_clean"] = df["text"].apply(clean_text)

    # drop rows where cleaning left almost nothing
    df = df[df["text_clean"].str.len() >= 10].reset_index(drop=True)

    out_path = input_path.with_name(input_path.stem + "_clean.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}  ({len(df):,} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean review text")
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    clean_dataset(args.input)


if __name__ == "__main__":
    main()
