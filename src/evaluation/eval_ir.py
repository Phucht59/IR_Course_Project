"""
eval_ir.py – Compute standard IR metrics from TREC-style qrels and run files.

Metrics: Precision@k, Recall@k, MAP, nDCG@k.
Supports both binary (0/1) and graded (0/1/2) relevance judgments.

File formats
------------
qrels  (space-separated, no header):  query_id  0  doc_id  relevance
run    (space-separated, no header):  query_id  Q0  doc_id  rank  score  run_name

Usage:
    # Evaluate with pseudo-qrels
    python src/evaluation/eval_ir.py \
        --qrels evaluation/qrels/qrels_pseudo.txt \
        --run   evaluation/runs/run_bm25.txt \
        --k 5 10

    # Evaluate with manual-qrels (graded 0/1/2)
    python src/evaluation/eval_ir.py \
        --qrels evaluation/qrels/qrels_manual.txt \
        --run   evaluation/runs/run_bm25.txt \
        --qrels-type manual \
        --k 5 10

    # Compare both systems at once:
    python src/evaluation/eval_ir.py \
        --qrels evaluation/qrels/qrels_pseudo.txt \
        --run evaluation/runs/run_tfidf.txt evaluation/runs/run_bm25.txt \
        --k 5 10
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation" / "reports"


# ── file loaders ────────────────────────────────────────────────────
def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}}.

    Supports graded relevance (0, 1, 2).
    """
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels[qid][did] = rel
    return dict(qrels)


def load_run(path: Path) -> dict[str, list[str]]:
    """Return {query_id: [doc_id, …]} in rank order."""
    run: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, did, rank = parts[0], parts[2], int(parts[3])
            run[qid].append((rank, did))
    return {qid: [did for _, did in sorted(docs)] for qid, docs in run.items()}


# ── metric implementations ─────────────────────────────────────────
def precision_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """Fraction of top-k documents that are relevant (rel > 0)."""
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """Fraction of all relevant documents retrieved in top-k."""
    total_rel = sum(1 for v in rel.values() if v > 0)
    if total_rel == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / total_rel


def average_precision(ranked: list[str], rel: dict[str, int]) -> float:
    """Mean of precision values at each relevant-document position."""
    total_rel = sum(1 for v in rel.values() if v > 0)
    if total_rel == 0:
        return 0.0
    hits, ap = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if rel.get(d, 0) > 0:
            hits += 1
            ap += hits / i
    return ap / total_rel


def dcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """Discounted Cumulative Gain using graded relevance."""
    return sum(
        (2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k])
    )


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """Normalized DCG — handles graded relevance (0/1/2)."""
    actual = dcg_at_k(ranked, rel, k)
    ideal_order = sorted(rel.values(), reverse=True)[:k]
    ideal = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_order))
    return actual / ideal if ideal > 0 else 0.0


# ── evaluate one run ───────────────────────────────────────────────
def evaluate_run(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    ks: list[int],
    run_name: str = "",
) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for qid in sorted(qrels):
        ranked = run.get(qid, [])
        rel = qrels[qid]
        row: dict[str, object] = {"query_id": qid}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(ranked, rel, k)
            row[f"Recall@{k}"] = recall_at_k(ranked, rel, k)
            row[f"nDCG@{k}"] = ndcg_at_k(ranked, rel, k)
        row["AP"] = average_precision(ranked, rel)
        rows.append(row)

    df = pd.DataFrame(rows)
    means = df.drop(columns=["query_id"]).mean()

    header = f"  Run: {run_name}" if run_name else ""
    print(f"\n{'='*60}{header}")
    print("Per-query metrics:")
    print(df.to_string(index=False))
    print(f"\nMean across {len(df)} queries:")
    for col, val in means.items():
        label = "MAP" if col == "AP" else col
        print(f"  {label:12s} = {val:.4f}")

    return df, means


# ── main ────────────────────────────────────────────────────────────
def evaluate(
    qrels_path: Path,
    run_paths: list[Path],
    ks: list[int],
    qrels_type: str = "pseudo",
):
    qrels = load_qrels(qrels_path)

    # Summarise qrels statistics
    all_rels = [r for doc_rels in qrels.values() for r in doc_rels.values()]
    grade_counts = {}
    for r in all_rels:
        grade_counts[r] = grade_counts.get(r, 0) + 1
    print(f"Qrels type  : {qrels_type}")
    print(f"Queries     : {len(qrels)}")
    print(f"Judgments   : {len(all_rels)}  (grade distribution: {grade_counts})")

    EVAL.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    suffix = f"_{qrels_type}" if qrels_type != "pseudo" else ""

    for run_path in run_paths:
        run = load_run(run_path)
        run_name = run_path.stem
        df, means = evaluate_run(qrels, run, ks, run_name)

        # save per-query
        df.to_csv(EVAL / f"ir_per_query_{run_name}{suffix}.csv", index=False)

        row = {"run": run_name}
        for col, val in means.items():
            label = "MAP" if col == "AP" else col
            row[label] = round(val, 4)
        summary_rows.append(row)

    # combined summary
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(EVAL / f"ir_metrics{suffix}.csv", index=False)
    print(f"\n{'='*60}")
    print("Summary comparison:")
    print(summary.to_string(index=False))
    print(f"\nSaved to {EVAL}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate IR run(s) against qrels")
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument(
        "--run",
        required=True,
        nargs="+",
        type=Path,
        help="One or more TREC-style run files",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10])
    parser.add_argument(
        "--qrels-type",
        choices=["pseudo", "manual"],
        default="pseudo",
        help="Label for the qrels source (affects output filenames, default: pseudo)",
    )
    args = parser.parse_args()
    evaluate(args.qrels, args.run, args.k, qrels_type=args.qrels_type)


if __name__ == "__main__":
    main()
