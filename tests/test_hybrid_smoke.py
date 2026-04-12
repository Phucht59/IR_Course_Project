"""
test_hybrid_smoke.py – Smoke test for the hybrid retrieval pipeline.

Verifies that imports, hybrid search, sentiment filtering, and
recommendation aggregation work end-to-end without crashing and
produce outputs with the expected shape.

Usage:
    python tests/test_hybrid_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_imports() -> None:
    """Verify that both modules import cleanly."""
    from indexing.indexing import build_dense_index        # noqa: F401
    from retrieval.retrieval import (                      # noqa: F401
        hybrid_search,
        filter_by_sentiment,
        recommend_restaurants,
        search_and_recommend,
    )
    print("[PASS] All imports succeeded.")


def test_hybrid_search() -> None:
    """Run a small hybrid search and check output shape."""
    from retrieval.retrieval import hybrid_search

    results = hybrid_search("best pizza fresh ingredients", top_k=20)

    assert not results.empty, "hybrid_search returned empty DataFrame"
    assert "rrf_score" in results.columns, "Missing 'rrf_score' column"
    assert "review_id" in results.columns, "Missing 'review_id' column"
    assert len(results) <= 20 * 2, "Result set unexpectedly large"  # RRF union <= 2*top_k
    print(f"[PASS] hybrid_search returned {len(results)} results.")


def test_filter_by_sentiment() -> None:
    """Filter results and verify size reduction."""
    from retrieval.retrieval import hybrid_search, filter_by_sentiment

    raw = hybrid_search("terrible food slow service", top_k=20)
    filtered = filter_by_sentiment(raw, threshold=-0.3)

    assert len(filtered) <= len(raw), "Filtered set should be <= raw set"
    assert "vader_compound" in filtered.columns, "Missing 'vader_compound' column"
    if not filtered.empty:
        assert (filtered["vader_compound"] > -0.3).all(), \
            "Some rows below threshold survived"
    print(f"[PASS] filter_by_sentiment: {len(raw)} -> {len(filtered)} reviews.")


def test_recommend_restaurants() -> None:
    """End-to-end: search -> filter -> recommend."""
    from retrieval.retrieval import search_and_recommend

    raw, filtered, recs = search_and_recommend(
        "great brunch mimosas", top_k=50, top_n=10,
    )

    assert len(recs) <= 10, "Should return at most 10 restaurants"
    if not recs.empty:
        assert "business_score" in recs.columns, "Missing 'business_score'"
        assert "top_reviews" in recs.columns, "Missing 'top_reviews'"
        for reviews_list in recs["top_reviews"]:
            assert len(reviews_list) <= 3, "At most 3 representative reviews"
    print(f"[PASS] recommend_restaurants: {len(recs)} restaurants returned.")


def test_empty_query_raises() -> None:
    """Ensure empty query raises ValueError."""
    from retrieval.retrieval import hybrid_search

    try:
        hybrid_search("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] Empty query raises ValueError.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    tests = [
        test_imports,
        test_empty_query_raises,
        test_hybrid_search,
        test_filter_by_sentiment,
        test_recommend_restaurants,
    ]
    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {test_fn.__name__}: {exc}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        sys.exit(1)
