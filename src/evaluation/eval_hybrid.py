"""
eval_hybrid.py – Chạy Hybrid Search trên tập query_set và xuất ra file TREC run.
"""

import argparse
import time
from pathlib import Path
import pandas as pd

from retrieval.retrieval import hybrid_search

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "evaluation" / "qrels" / "query_set.csv"
RUN_OUT_PATH = ROOT / "evaluation" / "runs" / "run_hybrid.txt"

def evaluate_hybrid(queries_path: Path, output_path: Path, top_k: int = 100, alpha: float = 0.9):
    print(f"Loading queries from {queries_path}...")
    queries_df = pd.read_csv(queries_path)
    
    run_lines = []
    
    start_time = time.time()
    for _, row in queries_df.iterrows():
        qid = row['query_id']
        qtext = row['query_text']
        
        print(f"Processing query {qid}: {qtext}")
        
        try:
            # Run hybrid search
            results_df = hybrid_search(qtext, top_k=top_k, alpha=alpha)
            
            # Format results into TREC run format: query_id Q0 doc_id rank score run_name
            for rank, (_, res_row) in enumerate(results_df.iterrows(), start=1):
                doc_id = res_row['review_id']
                # RRF score
                score = res_row['rrf_score']
                
                line = f"{qid} Q0 {doc_id} {rank} {score:.4f} hybrid_search"
                run_lines.append(line)
        except Exception as e:
            print(f"Error processing query {qid}: {e}")
            
    elapsed = time.time() - start_time
    print(f"Completed {len(queries_df)} queries in {elapsed:.2f}s")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(run_lines) + '\n')
    
    print(f"Saved {len(run_lines)} run lines to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--output", type=Path, default=RUN_OUT_PATH)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.9)
    
    args = parser.parse_args()
    evaluate_hybrid(args.queries, args.output, args.top_k, args.alpha)
