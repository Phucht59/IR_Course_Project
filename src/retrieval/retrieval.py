"""
retrieval.py – Hệ thống tìm kiếm lai (Hybrid retrieval) tích hợp bộ lọc cảm xúc
               và gợi ý nhà hàng sử dụng thuật toán Reciprocal Rank Fusion (RRF).

Luồng xử lý (Pipeline)
--------
1. ``hybrid_search``          – Kết hợp tìm kiếm BM25 + FAISS qua thuật toán RRF.
2. ``filter_by_sentiment``    – Loại bỏ các đánh giá dưới ngưỡng điểm VADER compound.
3. ``recommend_restaurants``  – Tổng hợp điểm cho nhà hàng và lấy Top-N nhà hàng
                                kèm theo các đánh giá tiêu biểu nhất.
4. ``search_and_recommend``   – Hàm bọc (wrapper) chạy từ A-Z một cách tiện lợi.

CLI
---
    python src/retrieval/retrieval.py --query "best pizza" --top-k 100
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# ── đường dẫn ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet"
BM25_PATH = ROOT / "models" / "bm25_index.pkl"
FAISS_PATH = ROOT / "models" / "faiss_index.bin"
DOC_IDS_PATH = ROOT / "models" / "dense_doc_ids.pkl"

RRF_K: int = 60  # Hằng số chuẩn cho thuật toán RRF

# ── singleton tải lười (lazy-loaded singletons) ──────────────────────────────────────────
_cache: dict[str, object] = {}


def _get_data() -> pd.DataFrame:
    """Tải và lưu cache DataFrame chứa dữ liệu đánh giá đã được làm sạch."""
    if "df" not in _cache:
        logger.info("Đang tải dataset từ %s …", DATA_PATH)
        _cache["df"] = pd.read_parquet(DATA_PATH)
    return _cache["df"]  # type: ignore[return-value]


def _get_bm25() -> BM25Okapi:
    """Tải và lưu cache index BM25."""
    if "bm25" not in _cache:
        logger.info("Đang tải BM25 index từ %s …", BM25_PATH)
        _cache["bm25"] = joblib.load(BM25_PATH)
    return _cache["bm25"]  # type: ignore[return-value]


def _get_faiss() -> faiss.Index:
    """Tải và lưu cache index FAISS."""
    if "faiss_index" not in _cache:
        logger.info("Đang tải FAISS index từ %s …", FAISS_PATH)
        _cache["faiss_index"] = faiss.read_index(str(FAISS_PATH))
    return _cache["faiss_index"]


def _get_doc_ids() -> list[str]:
    """Tải và lưu cache danh sách ID tài liệu đã sắp xếp cho FAISS."""
    if "doc_ids" not in _cache:
        with open(DOC_IDS_PATH, "rb") as fh:
            _cache["doc_ids"] = pickle.load(fh)
    return _cache["doc_ids"]  # type: ignore[return-value]


def _get_encoder() -> SentenceTransformer:
    """Tải và lưu cache mô hình mã hóa sentence-transformer."""
    if "encoder" not in _cache:
        logger.info("Đang tải mô hình mã hóa sentence-transformer …")
        _cache["encoder"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _cache["encoder"]  # type: ignore[return-value]


def _clean_query(text: str) -> str:
    """Viết thường, xóa bỏ ký tự không phải chữ/số, và chuẩn hóa dấu cách."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── 1. tìm kiếm lai (hybrid search) ───────────────────────────────────────────────
def hybrid_search(query: str, top_k: int = 100, alpha: float = 0.9) -> pd.DataFrame:
    """Truy xuất tài liệu qua BM25 + FAISS và trộn bằng RRF.

    Tham số
    ----------
    query : str
        Câu truy vấn văn bản tự do.
    top_k : int
        Số lượng ứng viên lấy ra từ *mỗi* hệ thống và là
        số lượng kết quả tối đa trả về sau khi trộn.

    Kết quả
    -------
    pd.DataFrame
        Kết quả đã xếp hạng bao gồm các cột ``review_id``,
        ``business_id``, ``text``, ``stars``, ``bm25_rank``,
        ``semantic_rank``, và ``rrf_score``.

    Lỗi
    ------
    ValueError
        Nếu *query* rỗng hoặc chỉ chứa khoảng trắng.
    """
    if not query or not query.strip():
        raise ValueError("Query phải là một chuỗi không rỗng.")

    t0 = time.perf_counter()

    df = _get_data()
    bm25 = _get_bm25()
    faiss_index = _get_faiss()
    encoder = _get_encoder()

    # ── Tìm kiếm BM25 ──────────────────────────────────────────────
    q_tokens: list[str] = _clean_query(query).split()
    bm25_scores: np.ndarray = bm25.get_scores(q_tokens)
    bm25_top_idx: np.ndarray = np.argsort(bm25_scores)[::-1][:top_k]

    bm25_ranks: dict[int, int] = {
        int(idx): rank for rank, idx in enumerate(bm25_top_idx, 1)
    }

    # ── Tìm kiếm ngữ nghĩa (Dense retrieval) ─────────────────────────────────────────────
    q_vec = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
    q_vec = normalize(q_vec, norm="l2")
    _, faiss_top_idx = faiss_index.search(q_vec, top_k)
    faiss_top_idx = faiss_top_idx[0]

    semantic_ranks: dict[int, int] = {
        int(idx): rank
        for rank, idx in enumerate(faiss_top_idx, 1)
        if idx >= 0  # FAISS trả về -1 cho các vị trí trống
    }

    # ── Trộn RRF ──────────────────────────────────────────────────
    all_candidates: set[int] = set(bm25_ranks) | set(semantic_ranks)
    rrf_rows: list[dict[str, object]] = []
    for idx in all_candidates:
        br = bm25_ranks.get(idx, top_k + 1)
        sr = semantic_ranks.get(idx, top_k + 1)
        rrf_score = alpha * (1.0 / (RRF_K + br)) + (1.0 - alpha) * (1.0 / (RRF_K + sr))
        rrf_rows.append(
            {
                "doc_idx": idx,
                "bm25_rank": br,
                "semantic_rank": sr,
                "rrf_score": rrf_score,
            }
        )

    rrf_df = (
        pd.DataFrame(rrf_rows)
        .sort_values("rrf_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # ── nối với dữ liệu tài liệu ─────────────────────────────────────
    result_indices: list[int] = rrf_df["doc_idx"].tolist()
    result = df.iloc[result_indices].reset_index(drop=True)
    result["bm25_rank"] = rrf_df["bm25_rank"].values
    result["semantic_rank"] = rrf_df["semantic_rank"].values
    result["rrf_score"] = rrf_df["rrf_score"].values

    cols = [
        "review_id", "business_id", "name", "text", "stars",
        "sentiment", "text_clean", "bm25_rank", "semantic_rank", "rrf_score",
    ]
    available = [c for c in cols if c in result.columns]
    result = result[available]

    elapsed = time.perf_counter() - t0
    logger.info(
        "hybrid_search: query=%r  -> %d kết quả trong %.2f s",
        query, len(result), elapsed,
    )
    return result


# ── 2. bộ lọc cảm xúc (sentiment filter) ────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()


def filter_by_sentiment(
    results_df: pd.DataFrame,
    threshold: float = -0.3,
) -> pd.DataFrame:
    """Xóa các đánh giá có điểm VADER compound thấp hơn hoặc bằng *threshold*.

    Cột mới ``vader_compound`` được thêm vào DataFrame trả về để
    các hàm phía sau có thể sử dụng điểm này trực tiếp.

    Tham số
    ----------
    results_df : pd.DataFrame
        Bắt buộc phải chứa cột ``text``.
    threshold : float
        Các đánh giá có điểm compound **<=** giá trị này sẽ bị loại bỏ.

    Kết quả
    -------
    pd.DataFrame
        Bản sao đã lọc kèm theo cột ``vader_compound``.
    """
    if results_df.empty:
        logger.warning("filter_by_sentiment: nhận DataFrame rỗng.")
        return results_df.assign(vader_compound=pd.Series(dtype=float))

    compounds: pd.Series = results_df["text"].apply(
        lambda t: _vader.polarity_scores(str(t))["compound"]
    )
    filtered = results_df.assign(vader_compound=compounds)
    filtered = filtered[filtered["vader_compound"] > threshold].reset_index(drop=True)

    logger.info(
        "filter_by_sentiment: %d -> %d reviews (threshold=%.2f)",
        len(results_df), len(filtered), threshold,
    )
    return filtered


# ── 3. tổng hợp gợi ý (recommendation aggregation) ──────────────────────────────────
def recommend_restaurants(
    filtered_df: pd.DataFrame,
    top_n: int = 10,
    reviews_per_biz: int = 3,
) -> pd.DataFrame:
    """Tổng hợp mảng đánh giá đã lọc vào 1 danh sách gợi ý nhà hàng xếp hạng.

    Mỗi nhà hàng nhận một điểm số tính bằng::

        business_score = Σ  rrf_score_i  ×  (1 + vader_compound_i)

    trong đó phép tổng chạy qua tất cả các bài review của nhà hàng đó trong *filtered_df*.

    Tham số
    ----------
    filtered_df : pd.DataFrame
        Đầu ra của ``filter_by_sentiment`` (bắt buộc chứa các cột ``rrf_score``,
        ``vader_compound``, ``business_id``, ``text``).
    top_n : int
        Số lượng nhà hàng top muốn trả về.
    reviews_per_biz : int
        Số lượng đánh giá tiêu biểu được hiển thị cho mỗi nhà hàng.

    Kết quả
    -------
    pd.DataFrame
        Các cột: ``business_id``, ``name``, ``business_score``,
        ``review_count``, ``avg_stars``, ``top_reviews``.
    """
    if filtered_df.empty:
        logger.warning("recommend_restaurants: nhận DataFrame rỗng.")
        return pd.DataFrame(
            columns=[
                "business_id", "name", "business_score",
                "review_count", "avg_stars", "top_reviews",
            ]
        )

    # kiểm tra an toàn: đảm bảo các cột cần thiết tồn tại
    required = {"business_id", "rrf_score", "vader_compound", "text"}
    missing = required - set(filtered_df.columns)
    if missing:
        raise KeyError(f"Thiếu các cột bắt buộc: {missing}")

    # điểm số có trọng số cho mỗi đánh giá
    df = filtered_df.copy()
    df["weighted_score"] = df["rrf_score"] * (1.0 + df["vader_compound"])

    # tổng hợp
    agg = (
        df.groupby("business_id")
        .agg(
            name=("name", "first"),
            business_score=("weighted_score", "sum"),
            review_count=("review_id", "count") if "review_id" in df.columns
                else ("text", "count"),
            avg_stars=("stars", "mean") if "stars" in df.columns
                else ("business_score", lambda _: float("nan")),
        )
        .reset_index()
    )
    
    # Nhiệm vụ 1: Log-Smoothed Average (điểm trung bình mượt bằng hàm Log)
    agg["business_score"] = (agg["business_score"] / agg["review_count"]) * np.log10(10 + agg["review_count"])
    agg = agg.sort_values("business_score", ascending=False).head(top_n).reset_index(drop=True)

    # móc top-N review tiêu biểu (điểm RRF cao nhất trong nhà hàng)
    top_reviews_map: dict[str, list[str]] = {}
    for biz_id in agg["business_id"]:
        biz_reviews = (
            df[df["business_id"] == biz_id]
            .sort_values("rrf_score", ascending=False)
            .head(reviews_per_biz)
        )
        top_reviews_map[biz_id] = biz_reviews["text"].tolist()

    agg["top_reviews"] = agg["business_id"].map(top_reviews_map)

    logger.info(
        "recommend_restaurants: %d nhà hàng được chấm điểm, trả về top %d.",
        df["business_id"].nunique(), len(agg),
    )
    return agg


# ── 4. hàm bọc end-to-end tiện lợi ──────────────────────────────────────
def search_and_recommend(
    query: str,
    top_k: int = 100,
    vader_threshold: float = -0.3,
    top_n: int = 10,
    reviews_per_biz: int = 3,
    alpha: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chạy toàn bộ luồng tìm kiếm lai -> lọc -> gợi ý.

    Tham số
    ----------
    query : str
        Câu truy vấn văn bản tự do.
    top_k : int
        Số lượng ứng viên gộp từ mỗi hệ thống tìm kiếm.
    vader_threshold : float
        Ngưỡng giới hạn điểm VADER để lọc đánh giá.
    top_n : int
        Số lượng nhà hàng top cần trả về.
    reviews_per_biz : int
        Số lượng đánh giá tiêu biểu ứng với mỗi nhà hàng.

    Kết quả
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(raw_results, filtered_results, recommendations)``
    """
    raw = hybrid_search(query, top_k=top_k, alpha=alpha)
    filtered = filter_by_sentiment(raw, threshold=vader_threshold)
    recs = recommend_restaurants(
        filtered, top_n=top_n, reviews_per_biz=reviews_per_biz,
    )
    return raw, filtered, recs


# ── CLI ─────────────────────────────────────────────────────────────
_SNIPPET_LEN = 120


def main() -> None:
    """Điểm vào CLI cho quy trình search lai và gợi ý."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Tìm kiếm theo hướng lai BM25+FAISS qua RRF, kèm bộ lọc cảm xúc, "
                    "và hệ thống gợi ý nhà hàng.",
    )
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--vader-threshold", type=float, default=-0.3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    raw, filtered, recs = search_and_recommend(
        query=args.query,
        top_k=args.top_k,
        vader_threshold=args.vader_threshold,
        top_n=args.top_n,
        alpha=args.alpha,
    )

    print(f"\n{'='*70}")
    print(f"Truy vấn        : \"{args.query}\"")
    print(f"Lấy ra          : {len(raw)} tài liệu")
    print(f"Sau khi lọc     : {len(filtered)} tài liệu (VADER > {args.vader_threshold})")
    print(f"{'='*70}")

    if recs.empty:
        print("\nKhông có nhà hàng nào khớp với truy vấn sau bước lọc cảm xúc.")
        return

    print(f"\n  Top-{args.top_n} Nhà Hàng Được Gợi Ý\n  {'—'*50}")
    for rank, (_, row) in enumerate(recs.iterrows(), 1):
        print(
            f"\n  #{rank}  {row['name']}\n"
            f"       điểm số={row['business_score']:.4f}  "
            f"đánh giá={row['review_count']}  "
            f"sao trung bình={row['avg_stars']:.1f}"
        )
        for i, rev in enumerate(row["top_reviews"], 1):
            snippet = str(rev)[:_SNIPPET_LEN].replace("\n", " ")
            ellipsis = "…" if len(str(rev)) > _SNIPPET_LEN else ""
            print(f"       [{i}] \"{snippet}{ellipsis}\"")


if __name__ == "__main__":
    main()
