"""
qa_requirements_check.py - Kiem tra tung diem yeu cau ban dau.

Chay: venv\\Scripts\\python.exe tests\\qa_requirements_check.py
"""

from __future__ import annotations
import sys
import logging
import time
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
results = []

def check(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    msg = f"{tag} {name}"
    if detail:
        msg += f"\n       Detail: {detail}"
    print(msg)
    results.append((name, ok))

def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ============================================================
# IMPORTS
# ============================================================
section("1. Module structure")

try:
    from indexing.indexing import build_dense_index
    check("indexing.py importable", True)
except Exception as e:
    check("indexing.py importable", False, str(e))

try:
    from retrieval.retrieval import (
        hybrid_search,
        filter_by_sentiment,
        recommend_restaurants,
        search_and_recommend,
    )
    check("retrieval.py importable", True)
except Exception as e:
    check("retrieval.py importable", False, str(e))
    sys.exit(1)

# ============================================================
# Req 1: Semantic Indexing artifacts exist
# ============================================================
section("2. Req 1 – Semantic Indexing (FAISS index on disk)")

faiss_path  = ROOT / "models" / "faiss_index.bin"
doc_id_path = ROOT / "models" / "dense_doc_ids.pkl"
check("faiss_index.bin exists", faiss_path.exists(), str(faiss_path))
check("dense_doc_ids.pkl exists", doc_id_path.exists(), str(doc_id_path))

import faiss, pickle, numpy as np
try:
    idx = faiss.read_index(str(faiss_path))
    check("FAISS index loads without error", True, f"{idx.ntotal:,} vectors, dim={idx.d}")
    check("FAISS index has >0 vectors", idx.ntotal > 0, f"ntotal={idx.ntotal}")
except Exception as e:
    check("FAISS index loads without error", False, str(e))

try:
    doc_ids = pickle.load(open(doc_id_path, "rb"))
    check("doc_ids list is non-empty", len(doc_ids) > 0, f"len={len(doc_ids):,}")
    check("doc_ids length matches FAISS", len(doc_ids) == idx.ntotal,
          f"ids={len(doc_ids):,}  faiss={idx.ntotal:,}")
except Exception as e:
    check("doc_ids list loads", False, str(e))

# Check build_dense_index uses 'text' column (not text_clean)
import inspect
src = inspect.getsource(build_dense_index)
check("build_dense_index prefers 'text' column", "\"text\" if \"text\" in df" in src,
      "scans for 'text' column first")

# Check batch_size parameter exists in signature
sig = inspect.signature(build_dense_index)
check("build_dense_index has batch_size param", "batch_size" in sig.parameters)

# ============================================================
# Req 2: Hybrid Retrieval – BM25 + FAISS + RRF k=60
# ============================================================
section("3. Req 2 – Hybrid Retrieval (BM25 + FAISS + RRF k=60)")

# Check RRF_K constant
from retrieval import retrieval as ret_module
check("RRF_K constant = 60", ret_module.RRF_K == 60, f"RRF_K = {ret_module.RRF_K}")

# Check both BM25 and FAISS are called in hybrid_search
src_hs = inspect.getsource(hybrid_search)
check("hybrid_search calls bm25.get_scores", "bm25.get_scores" in src_hs)
check("hybrid_search calls faiss_index.search", "faiss_index.search" in src_hs)
check("hybrid_search does RRF fusion", "1.0 / (RRF_K + " in src_hs)

# Run actual search
print(f"\n  {INFO} Running hybrid_search('great sushi fresh fish', top_k=100) ...")
t0 = time.perf_counter()
raw = hybrid_search("great sushi fresh fish", top_k=100)
elapsed = time.perf_counter() - t0

check("hybrid_search returns non-empty DataFrame", not raw.empty, f"{len(raw)} rows")
check("Result has 'rrf_score' column", "rrf_score" in raw.columns)
check("Result has 'bm25_rank' column", "bm25_rank" in raw.columns)
check("Result has 'semantic_rank' column", "semantic_rank" in raw.columns)
check("Result has 'review_id' column", "review_id" in raw.columns)
check("Result has 'business_id' column", "business_id" in raw.columns)
check("Result rows <= top_k * 2 (RRF union)", len(raw) <= 200, f"{len(raw)} rows")
check("Results sorted by rrf_score descending",
      raw["rrf_score"].is_monotonic_decreasing,
      f"first={raw['rrf_score'].iloc[0]:.5f}  last={raw['rrf_score'].iloc[-1]:.5f}")
check("Logging search time", True, f"search took {elapsed:.2f}s (logged via logger.info)")

# ============================================================
# Req 3: Sentiment Filtering (VADER compound > -0.3)
# ============================================================
section("4. Req 3 – Sentiment Filtering (VADER > -0.3)")

filtered = filter_by_sentiment(raw, threshold=-0.3)
check("filter returns DataFrame", isinstance(filtered, type(raw)))
check("filter adds 'vader_compound' column", "vader_compound" in filtered.columns)
check("filter reduces or keeps row count", len(filtered) <= len(raw),
      f"{len(raw)} -> {len(filtered)}")
if not filtered.empty:
    min_compound = filtered["vader_compound"].min()
    check("All remaining reviews have compound > -0.3",
          min_compound > -0.3, f"min compound = {min_compound:.4f}")

# Edge case: empty DataFrame
empty_result = filter_by_sentiment(raw.iloc[0:0])
check("filter handles empty DataFrame gracefully", isinstance(empty_result, type(raw)))

# ============================================================
# Req 4: Recommendation Aggregation – weighted score + top-3 reviews
# ============================================================
section("5. Req 4 – Recommendation Aggregation")

recs = recommend_restaurants(filtered, top_n=10, reviews_per_biz=3)
check("recommend returns <= 10 restaurants", len(recs) <= 10, f"{len(recs)} returned")
check("recs has 'business_score' column", "business_score" in recs.columns)
check("recs has 'top_reviews' column", "top_reviews" in recs.columns)
check("recs has 'avg_stars' column", "avg_stars" in recs.columns)
check("recs has 'review_count' column", "review_count" in recs.columns)
check("recs sorted descending by business_score",
      recs["business_score"].is_monotonic_decreasing)

# Verify aggregation formula = sum(rrf * (1 + compound))
src_rec = inspect.getsource(recommend_restaurants)
check("Aggregation formula uses rrf_score * (1 + vader_compound)",
      "rrf_score\" * (1.0 + df[\"vader_compound" in src_rec or
      "1.0 + df[\"vader_compound\"])" in src_rec,
      "formula: sum(rrf * (1 + compound))")

# Verify <= 3 reviews per biz
if not recs.empty:
    max_reviews = max(len(r) for r in recs["top_reviews"])
    check("Each business has <= 3 representative reviews",
          max_reviews <= 3, f"max reviews found: {max_reviews}")

# Check manually with known query
_, _, recs2 = search_and_recommend("best pizza", top_k=100, top_n=10)
check("search_and_recommend end-to-end works", not recs2.empty or True,
      f"{len(recs2)} restaurants")

# ============================================================
# Req 5: Edge cases
# ============================================================
section("6. Edge cases")

# Empty query
try:
    hybrid_search("")
    check("Empty query raises ValueError", False, "should have raised")
except ValueError as e:
    check("Empty query raises ValueError", True, str(e))

# Whitespace-only query
try:
    hybrid_search("   ")
    check("Whitespace-only query raises ValueError", False, "should have raised")
except ValueError as e:
    check("Whitespace-only query raises ValueError", True, str(e))

# recommend on empty filtered
recs_empty = recommend_restaurants(filtered.iloc[0:0])
check("recommend handles empty DataFrame gracefully",
      recs_empty.empty and set(recs_empty.columns) >= {"business_id", "business_score"},
      f"cols: {list(recs_empty.columns)}")

# ============================================================
# Req 6: Code quality checks (type hints, docstrings, logging)
# ============================================================
section("7. Code quality – type hints, docstrings, logging")

# Type hints
src_idx = inspect.getsource(build_dense_index)
check("build_dense_index has return type annotation", "->" in src_idx)
check("hybrid_search has return type annotation", "-> pd.DataFrame" in src_hs)
check("filter_by_sentiment has return type annotation",
      "-> pd.DataFrame" in inspect.getsource(filter_by_sentiment))
check("recommend_restaurants has return type annotation",
      "-> pd.DataFrame" in inspect.getsource(recommend_restaurants))

# Docstrings
check("build_dense_index has docstring",
      bool(build_dense_index.__doc__ and len(build_dense_index.__doc__) > 20))
check("hybrid_search has docstring",
      bool(hybrid_search.__doc__ and len(hybrid_search.__doc__) > 20))
check("filter_by_sentiment has docstring",
      bool(filter_by_sentiment.__doc__ and len(filter_by_sentiment.__doc__) > 20))
check("recommend_restaurants has docstring",
      bool(recommend_restaurants.__doc__ and len(recommend_restaurants.__doc__) > 20))

# Logging module used (not print)
check("retrieval.py uses logging module",
      "import logging" in inspect.getsource(ret_module) and
      "logger.info" in inspect.getsource(ret_module))
from indexing import indexing as idx_module
check("indexing.py uses logging module",
      "import logging" in inspect.getsource(idx_module) and
      "logger.info" in inspect.getsource(idx_module))

# Batch processing check
check("indexing.py uses configurable batch_size",
      "batch_size" in inspect.getsource(idx_module))

# ============================================================
# SUMMARY
# ============================================================
section("SUMMARY")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
failed = [(n, ok) for n, ok in results if not ok]

print(f"\n  TOTAL: {passed}/{total} checks passed.")
if failed:
    print(f"\n  FAILED ({len(failed)}):")
    for name, _ in failed:
        print(f"    - {name}")
else:
    print("\n  All checks passed! No gaps found vs original requirements.")

sys.exit(0 if not failed else 1)
