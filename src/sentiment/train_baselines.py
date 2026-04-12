"""
train_baselines.py – Train Naive Bayes, Logistic Regression, Linear SVM on
review sentiment.

Supports two experiment modes:
  three_class : positive / neutral / negative  (default)
  binary      : positive / negative  (drops stars == 3)

Optional --balanced flag enables class_weight="balanced" on LR and SVC.

Usage:
    # 3-class (default, same as before)
    python src/sentiment/train_baselines.py \
        --input data/processed/subset_restaurants_20000_clean.parquet

    # binary sentiment
    python src/sentiment/train_baselines.py \
        --input data/processed/subset_restaurants_20000_clean.parquet \
        --mode binary

    # binary + class balancing
    python src/sentiment/train_baselines.py \
        --input data/processed/subset_restaurants_20000_clean.parquet \
        --mode binary --balanced
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
EVAL = ROOT / "evaluation" / "reports"


# ── label helpers ───────────────────────────────────────────────────
def assign_labels(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Assign sentiment labels from star ratings.

    three_class : stars <= 2 -> negative, == 3 -> neutral, >= 4 -> positive
    binary      : stars <= 2 -> negative, >= 4 -> positive  (drop 3-star)
    """
    df = df.copy()
    if mode == "binary":
        df = df[df["stars"] != 3].copy()
        df["sentiment"] = df["stars"].apply(
            lambda s: "negative" if s <= 2 else "positive"
        )
    else:  # three_class

        def _label(s):
            if s <= 2:
                return "negative"
            elif s == 3:
                return "neutral"
            else:
                return "positive"

        df["sentiment"] = df["stars"].apply(_label)
    return df.reset_index(drop=True)


# ── training ────────────────────────────────────────────────────────
def train(
    input_path: Path,
    mode: str = "three_class",
    balanced: bool = False,
    test_size: float = 0.2,
    seed: int = 42,
    output_dir: Path | None = None,
):
    out = output_dir or EVAL
    tag = f"{mode}" + ("_balanced" if balanced else "")

    print(f"Reading {input_path} …")
    df = pd.read_parquet(input_path)

    # Apply labeling scheme
    df = assign_labels(df, mode)
    labels = sorted(df["sentiment"].unique())

    print(f"Mode: {mode}  |  Balanced: {balanced}  |  Labels: {labels}")
    print(f"Class distribution:\n{df['sentiment'].value_counts().to_string()}\n")

    X = df["text_clean"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    print(f"Train {len(X_train):,}  |  Test {len(X_test):,}")

    # shared vectorizer
    vec = TfidfVectorizer(max_features=40_000, sublinear_tf=True, ngram_range=(1, 2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # build classifiers
    cw = "balanced" if balanced else None
    classifiers = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight=cw,
        ),
        "LinearSVC": LinearSVC(
            max_iter=2000,
            random_state=seed,
            class_weight=cw,
        ),
    }

    best_name, best_f1, best_model = None, -1, None
    summary_rows = []

    for name, clf in classifiers.items():
        print(f"\nTraining {name} …")
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        mf1 = f1_score(y_test, preds, average="macro", zero_division=0)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro-F1 : {mf1:.4f}")

        report = classification_report(y_test, preds, labels=labels, zero_division=0)
        print(report)

        cm = confusion_matrix(y_test, preds, labels=labels)
        print("  Confusion matrix:")
        print(f"  {cm}\n")

        summary_rows.append({"model": name, "accuracy": acc, "macro_f1": mf1})

        if mf1 > best_f1:
            best_f1 = mf1
            best_name = name
            best_model = clf

    # persist best model + vectorizer
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS / f"sentiment_best_model_{tag}.pkl")
    joblib.dump(vec, MODELS / f"sentiment_vectorizer_{tag}.pkl")
    print(f"\n* Best model: {best_name} (Macro-F1 = {best_f1:.4f}) -> saved")

    # save predictions for eval_sentiment.py
    best_preds = best_model.predict(X_test_vec)
    pred_df = pd.DataFrame(
        {
            "review_id": df.loc[X_test.index, "review_id"].values,
            "y_true": y_test.values,
            "y_pred": best_preds,
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out / f"sentiment_predictions_{tag}.csv", index=False)

    # classification report CSV
    report_dict = classification_report(
        y_test,
        best_preds,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(out / f"classification_report_{tag}.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / f"sentiment_summary_{tag}.csv", index=False)

    # print comparison table
    print(f"\n{'='*60}")
    print(f"Comparison table  ({tag}):")
    print(summary_df.to_string(index=False))
    print(f"\nPredictions & summary saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Train sentiment baselines")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["three_class", "binary"],
        default="three_class",
        help="Labeling scheme (default: three_class)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use class_weight='balanced' for LR and LinearSVC",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: evaluation/reports)",
    )
    args = parser.parse_args()

    train(
        args.input,
        mode=args.mode,
        balanced=args.balanced,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
