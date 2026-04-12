"""
build_dense.py – Build a FAISS dense vector index using sentence-transformers.

Encodes every document (clean_text column) with a sentence-transformer model,
L2-normalises the vectors for cosine similarity, and stores a FAISS IndexFlatIP
index plus the ordered list of review_ids.

Usage:
    python src/indexing/build_dense.py
    python src/indexing/build_dense.py --model-name all-MiniLM-L6-v2
"""

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet"
MODELS = ROOT / "models"


def build(model_name: str = "all-MiniLM-L6-v2") -> None:
    # ── load data ───────────────────────────────────────────────────
    print(f"[1/4] Reading {DATA_PATH} …")
    df = pd.read_parquet(DATA_PATH)
    texts = df["text_clean"].fillna("").tolist()
    doc_ids = df["review_id"].tolist()
    print(f"       {len(texts):,} documents loaded.")

    # ── encode ──────────────────────────────────────────────────────
    print(f"[2/4] Encoding with '{model_name}' (batch_size=64) …")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    print(f"       Embedding matrix shape: {embeddings.shape}")

    # ── normalise for cosine similarity ─────────────────────────────
    print("[3/4] L2-normalising vectors …")
    faiss.normalize_L2(embeddings)

    # ── build FAISS index ───────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"       FAISS index built – {index.ntotal:,} vectors, dim={dim}")

    # ── save ────────────────────────────────────────────────────────
    MODELS.mkdir(parents=True, exist_ok=True)

    faiss_path = MODELS / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    print(f"[4/4] Saved FAISS index  -> {faiss_path}")

    ids_path = MODELS / "dense_doc_ids.pkl"
    with open(ids_path, "wb") as f:
        pickle.dump(doc_ids, f)
    print(f"       Saved doc-id list -> {ids_path}")

    print("\nDone [OK]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS dense index with sentence-transformers"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()
    build(model_name=args.model_name)


if __name__ == "__main__":
    main()
