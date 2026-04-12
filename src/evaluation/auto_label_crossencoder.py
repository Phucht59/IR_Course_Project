"""
auto_label_crossencoder.py – Dán nhãn liên quan (Relevance Judgment) bằng mô hình 
Cross-Encoder pre-trained, thay thế phương pháp từ điển đồng nghĩa thủ công.

CƠ SỞ KHOA HỌC:
  Phương pháp này dựa trên nghiên cứu của Faggioli et al. (2023):
  "Perspectives on Large Language Models for Relevance Judgment" (SIGIR 2023),
  trong đó chứng minh rằng mô hình ngôn ngữ lớn (LLM) và Cross-Encoder có thể
  thay thế con người (human assessors) trong vai trò thẩm định viên liên quan
  với mức độ đồng thuận (inter-annotator agreement) tương đương giữa người với người.

  Cụ thể, đồ án sử dụng mô hình Cross-Encoder 'cross-encoder/ms-marco-MiniLM-L-6-v2'
  được huấn luyện trên tập MS MARCO (500k+ cặp query-passage đánh giá bởi con người)
  để chấm điểm liên quan. Mô hình này:
  - Nhận đầu vào: (query, document_text)
  - Trả về: điểm số liên tục thể hiện mức độ liên quan
  - Đã được kiểm chứng trên nhiều benchmark IR quốc tế (TREC DL, BEIR)

  Phân bậc liên quan (Graded Relevance):
    score >= 4.0  → label 2 (Highly Relevant)
    score >= 1.0  → label 1 (Partially Relevant)
    score <  1.0  → label 0 (Not Relevant)

  Ngưỡng được xác định dựa trên phân phối điểm thực tế của MS MARCO Cross-Encoder.

Usage:
    python src/evaluation/auto_label_crossencoder.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL_PATH = ROOT / "evaluation" / "qrels" / "human_label_pool.csv"
QRELS_OUT = ROOT / "evaluation" / "qrels" / "qrels_crossencoder.txt"
DATA_PATH = ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet"


def main():
    print("Loading Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    # Load pool
    pool = pd.read_csv(POOL_PATH)
    print(f"Loaded pool: {len(pool)} pairs, {pool['query_id'].nunique()} queries")

    # Load full review text (snippet may be truncated)
    data = pd.read_parquet(DATA_PATH, columns=["review_id", "text"])
    data = data.rename(columns={"review_id": "doc_id", "text": "full_text"})
    pool = pool.merge(data, on="doc_id", how="left")

    # Prepare pairs for Cross-Encoder
    pairs = []
    for _, row in pool.iterrows():
        query = str(row["query_text"])
        doc = str(row.get("full_text", row.get("review_snippet", "")))
        # Truncate doc to 400 chars to fit model context
        doc = doc[:400]
        pairs.append((query, doc))

    print(f"Scoring {len(pairs)} pairs with Cross-Encoder...")
    scores = model.predict(pairs, show_progress_bar=True)

    # Assign graded labels based on score thresholds
    pool["ce_score"] = scores
    pool["label"] = pool["ce_score"].apply(
        lambda s: 2 if s >= 4.0 else (1 if s >= 1.0 else 0)
    )

    # Print distribution
    dist = pool["label"].value_counts().sort_index()
    print(f"\nLabel distribution:")
    for label, count in dist.items():
        pct = count / len(pool) * 100
        names = {0: "Not Relevant", 1: "Partially Relevant", 2: "Highly Relevant"}
        print(f"  {label} ({names[label]:>20s}): {count:4d} ({pct:.1f}%)")

    print(f"\nScore statistics:")
    print(f"  Mean:   {pool['ce_score'].mean():.3f}")
    print(f"  Median: {pool['ce_score'].median():.3f}")
    print(f"  Min:    {pool['ce_score'].min():.3f}")
    print(f"  Max:    {pool['ce_score'].max():.3f}")

    # Write TREC qrels
    qrels_lines = []
    for _, row in pool.iterrows():
        qrels_lines.append(f"{row['query_id']} 0 {row['doc_id']} {row['label']}")

    QRELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    QRELS_OUT.write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")
    print(f"\nSaved TREC qrels -> {QRELS_OUT}")

    # Also save the scored pool for reference
    scored_path = POOL_PATH.parent / "human_label_pool_scored.csv"
    pool.to_csv(scored_path, index=False)
    print(f"Saved scored pool -> {scored_path}")

    n_rel = sum(1 for l in pool["label"] if l > 0)
    print(f"\nTotal: {len(pool)} judgments, {n_rel} relevant (label > 0)")


if __name__ == "__main__":
    main()
