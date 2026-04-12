"""
build_compact_label_pool.py – Tạo tập dán nhãn thủ công gọn nhất cho đánh giá IR.

Chiến lược:
  - Pool Top-10 từ mỗi hệ thống (BM25, TF-IDF, Hybrid) cho mỗi query
  - Loại trùng lặp (union-dedup)
  - Xuất CSV ngắn gọn.
  - Con người (sinh viên) đọc review_text rồi dán nhãn:
      0 = Không liên quan  
      1 = Liên quan một phần (nhắc đến chủ đề nhưng không trọng tâm)
      2 = Rất liên quan (nói trực tiếp về chủ đề truy vấn)

Cơ sở khoa học:
  - Phương pháp Pooling chuẩn TREC (Voorhees, 2000):
    "The pool is formed by taking the top-k documents from each participating
    system and judging only those documents."
  - Con người là Gold Standard thẩm định viên (Human Assessor)
    → Kết quả đánh giá không phụ thuộc vào từ điển đồng nghĩa tự xây.

Usage:
    python src/evaluation/build_compact_label_pool.py
"""

from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet"
QUERY_PATH = ROOT / "evaluation" / "qrels" / "query_set.csv"
RUNS_DIR = ROOT / "evaluation" / "runs"
OUTPUT_PATH = ROOT / "evaluation" / "qrels" / "human_label_pool.csv"

POOL_DEPTH = 10  # Top-10 per system per query → đảm bảo mọi bài trong @10 đều có nhãn


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


def main():
    queries = pd.read_csv(QUERY_PATH)
    data = pd.read_parquet(
        DATA_PATH,
        columns=["review_id", "name", "stars", "text_clean", "text"],
    )

    # Load all available run files
    run_files = list(RUNS_DIR.glob("run_*.txt"))
    print(f"Found {len(run_files)} run files: {[f.name for f in run_files]}")

    runs = {}
    for rf in run_files:
        system_name = rf.stem.replace("run_", "")
        runs[system_name] = load_run_topk(rf, POOL_DEPTH)

    pool_rows = []
    for _, qrow in queries.iterrows():
        qid = qrow["query_id"]
        qtxt = qrow["query_text"]
        aspect = qrow["aspect"]

        # Collect docs from all systems
        seen = set()
        for sys_name, sys_run in runs.items():
            for doc_id in sys_run.get(qid, []):
                if doc_id not in seen:
                    seen.add(doc_id)
                    # Find which systems returned this doc
                    sources = [
                        s for s, r in runs.items() if doc_id in r.get(qid, [])
                    ]
                    pool_rows.append({
                        "query_id": qid,
                        "aspect": aspect,
                        "query_text": qtxt,
                        "doc_id": doc_id,
                        "source_systems": "+".join(sources),
                    })

    pool = pd.DataFrame(pool_rows)

    # Join review metadata
    data_lookup = data.rename(columns={
        "review_id": "doc_id",
        "name": "business_name",
        "text": "review_text_original",
        "text_clean": "review_text_clean",
    })
    pool = pool.merge(data_lookup, on="doc_id", how="left")

    # Truncate review text for easier reading (first 300 chars)
    pool["review_snippet"] = pool["review_text_original"].str[:300]

    # Add blank annotation columns
    pool["label"] = ""
    pool["notes"] = ""

    # Final column order
    col_order = [
        "query_id", "aspect", "query_text",
        "doc_id", "business_name", "stars",
        "review_snippet", "source_systems",
        "label", "notes",
    ]
    pool = pool[[c for c in col_order if c in pool.columns]]

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Saved human label pool -> {OUTPUT_PATH}")
    print(f"  Queries          : {pool['query_id'].nunique()}")
    print(f"  Total pairs      : {len(pool)}")
    print(f"  Avg docs/query   : {len(pool) / pool['query_id'].nunique():.1f}")
    print(f"\n  HƯỚNG DẪN DÁN NHÃN:")
    print(f"  Mở file CSV, đọc cột 'review_snippet' rồi điền vào cột 'label':")
    print(f"    0 = Không liên quan (bài review nói về chủ đề khác)")
    print(f"    1 = Liên quan một phần (có nhắc đến nhưng không phải trọng tâm)")
    print(f"    2 = Rất liên quan (nói trực tiếp và đầy đủ về chủ đề truy vấn)")


if __name__ == "__main__":
    main()
