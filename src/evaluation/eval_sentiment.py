"""
eval_sentiment.py – Produce detailed sentiment evaluation artefacts from
the predictions CSV exported by train_baselines.py.

Usage:
    python src/evaluation/eval_sentiment.py
    python src/evaluation/eval_sentiment.py --preds evaluation/reports/sentiment_predictions.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation" / "reports"


def evaluate(preds_path: Path):
    df = pd.read_csv(preds_path)
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    labels = ["negative", "neutral", "positive"]

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {mf1:.4f}\n")

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    report_df = pd.DataFrame(report).T
    print(report_df.to_string())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(f"\nConfusion matrix:\n{cm_df.to_string()}")

    # save artefacts
    EVAL.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(EVAL / "classification_report.csv")
    cm_df.to_csv(EVAL / "confusion_matrix.csv")
    print(f"\nSaved to {EVAL}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment predictions")
    parser.add_argument(
        "--preds",
        type=Path,
        default=EVAL / "sentiment_predictions.csv",
    )
    args = parser.parse_args()
    evaluate(args.preds)


if __name__ == "__main__":
    main()
