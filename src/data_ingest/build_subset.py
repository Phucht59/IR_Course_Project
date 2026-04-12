"""
build_subset.py – Extract a manageable subset from Yelp JSON dumps.

Usage:
    python src/data_ingest/build_subset.py --category Restaurants --n 20000
    python src/data_ingest/build_subset.py --category Restaurants --city "Las Vegas" --n 50000
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# ── defaults ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

REVIEW_FILE = RAW / "yelp_academic_dataset_review.json"
BUSINESS_FILE = RAW / "yelp_academic_dataset_business.json"


def load_jsonl(path: Path, cols: list[str] | None = None) -> pd.DataFrame:
    """Read a line-delimited JSON file into a DataFrame."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if cols:
                obj = {k: obj.get(k) for k in cols}
            records.append(obj)
    return pd.DataFrame(records)


def map_sentiment(stars: int) -> str:
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def build_subset(
    category: str = "Restaurants",
    city: str | None = None,
    n: int = 20_000,
    seed: int = 42,
    min_text_len: int = 30,
    out_dir: Path = OUT,
) -> pd.DataFrame:
    print(f"[1/5] Loading business file …")
    biz_cols = ["business_id", "name", "categories", "city", "state", "stars"]
    biz = load_jsonl(BUSINESS_FILE, biz_cols)

    # filter by category
    mask = biz["categories"].fillna("").str.contains(category, case=False)
    biz = biz[mask]
    if city:
        biz = biz[biz["city"].str.lower() == city.lower()]
    print(f"   businesses after filter: {len(biz):,}")

    print(f"[2/5] Loading review file …")
    rev_cols = ["review_id", "business_id", "stars", "text", "date"]
    reviews = load_jsonl(REVIEW_FILE, rev_cols)

    print(f"[3/5] Merging & cleaning …")
    merged = reviews.merge(
        biz[["business_id", "name", "categories", "city"]],
        on="business_id",
        how="inner",
        suffixes=("", "_biz"),
    )
    # rename review-level stars (keep only review stars)
    if "stars_biz" in merged.columns:
        merged.drop(columns=["stars_biz"], inplace=True)

    # drop null / short text
    merged = merged.dropna(subset=["text"])
    merged = merged[merged["text"].str.len() >= min_text_len]

    # sample
    if n and n < len(merged):
        merged = merged.sample(n=n, random_state=seed)
    merged = merged.reset_index(drop=True)

    # sentiment label
    merged["sentiment"] = merged["stars"].apply(map_sentiment)

    print(f"[4/5] Saving to {out_dir} …")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{category.lower()}"
    if city:
        tag += f"_{city.lower().replace(' ', '_')}"
    tag += f"_{len(merged)}"
    out_path = out_dir / f"subset_{tag}.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"   saved -> {out_path}")

    # summary
    print(f"\n[5/5] Summary")
    print(f"   reviews : {len(merged):,}")
    print(f"   businesses : {merged['business_id'].nunique():,}")
    print(f"\n   Sentiment distribution:")
    print(merged["sentiment"].value_counts().to_string())
    print(f"\n   Top 5 cities:")
    print(merged["city"].value_counts().head().to_string())
    print(f"\n   Top 10 category strings:")
    print(merged["categories"].value_counts().head(10).to_string())

    return merged


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build Yelp subset")
    parser.add_argument("--category", default="Restaurants")
    parser.add_argument("--city", default=None)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=30)
    args = parser.parse_args()

    build_subset(
        category=args.category,
        city=args.city,
        n=args.n,
        seed=args.seed,
        min_text_len=args.min_len,
    )


if __name__ == "__main__":
    main()
