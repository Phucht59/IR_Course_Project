"""
indexing.py – Build FAISS dense vector index using sentence-transformers.
"""

import argparse
import pickle
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet"
MODELS = ROOT / "models"


def build_dense_index(model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> None:
    """
    Encodes every document with a sentence-transformer model,
    L2-normalises the vectors for cosine similarity, and stores a FAISS index plus doc IDs.
    """
    logger.info(f"[1/4] Reading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    text_col = "text" if "text" in df else "text_clean"
    texts = df[text_col].fillna("").tolist()
    doc_ids = df["review_id"].tolist()
    logger.info(f"       {len(texts):,} documents loaded.")

    logger.info(f"[2/4] Encoding with '{model_name}' (batch_size={batch_size}) ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    logger.info(f"       Embedding matrix shape: {embeddings.shape}")

    logger.info("[3/4] L2-normalising vectors ...")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"       FAISS index built - {index.ntotal:,} vectors, dim={dim}")

    MODELS.mkdir(parents=True, exist_ok=True)

    faiss_path = MODELS / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"[4/4] Saved FAISS index  -> {faiss_path}")

    ids_path = MODELS / "dense_doc_ids.pkl"
    with open(ids_path, "wb") as f:
        pickle.dump(doc_ids, f)
    logger.info(f"       Saved doc-id list -> {ids_path}")

    logger.info("Done [OK]")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Build FAISS dense index with sentence-transformers"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding calculation",
    )
    args = parser.parse_args()
    build_dense_index(model_name=args.model_name, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
