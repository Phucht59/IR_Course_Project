"""
build_aspect_profiles.py – Tính toán cấu hình ABSA ngoại tuyến (offline) cho tất cả nhà hàng.

Chạy khâu nhận diện ABSA dựa trên luật tự định nghĩa qua toàn bộ các review đánh giá,
tổng hợp các đặc trưng theo nhà hàng, và lưu trữ cấu hình dưới định dạng Parquet + CSV.

Sử dụng:
    python src/analytics/build_aspect_profiles.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# cho phép nạp thư viện từ danh mục gốc của dự án
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.absa_rules import detect_aspects_in_text  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet"
EVAL_REPORTS = ROOT / "evaluation" / "reports"


def build_profiles(input_path: Path | None = None) -> pd.DataFrame:
    path = input_path or DATA_PATH

    print(f"[1/3] Đang đọc file {path} …")
    df = pd.read_parquet(path)
    print(f"       Tải xong {len(df):,} reviews.")

    # ── phân tích ABSA cho từng đánh giá ─────────────────────────────────────────────
    print("[2/3] Bắt đầu chấm điểm nhận diện khía cạnh trên toàn bộ tập review …")
    try:
        from tqdm import tqdm
        iterator = tqdm(df["text_clean"].fillna(""), total=len(df), desc="Tiến trình ABSA")
    except ImportError:
        print("       (vui lòng cài tqdm nếu muốn thấy thanh tiến trình)")
        iterator = df["text_clean"].fillna("")

    food_scores, service_scores, ambience_scores, price_scores = [], [], [], []

    for text in iterator:
        scores = detect_aspects_in_text(text)
        food_scores.append(scores["food"])
        service_scores.append(scores["service"])
        ambience_scores.append(scores["ambience"])
        price_scores.append(scores["price"])

    df["food_score"] = food_scores
    df["service_score"] = service_scores
    df["ambience_score"] = ambience_scores
    df["price_score"] = price_scores

    # ── tổng hợp điểm theo mỗi nhà hàng ──────────────────────────────────────
    print("[3/3] Tiến hành tổng hợp (Aggregate) điểm dựa vào ID nhà hàng …")
    profiles = (
        df.groupby("business_id")
        .agg(
            name=("name", "first"),
            review_count=("review_id", "count"),
            avg_food=("food_score", "mean"),
            avg_service=("service_score", "mean"),
            avg_ambience=("ambience_score", "mean"),
            avg_price=("price_score", "mean"),
            avg_stars=("stars", "mean"),
            positive_rate=(
                "sentiment",
                lambda s: (s == "positive").mean(),
            ),
            negative_rate=(
                "sentiment",
                lambda s: (s == "negative").mean(),
            ),
        )
        .reset_index()
    )

    # ── lưu lại file ────────────────────────────────────────────────────────
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)

    pq_path = EVAL_REPORTS / "aspect_profiles.parquet"
    csv_path = EVAL_REPORTS / "aspect_profiles.csv"

    profiles.to_parquet(pq_path, index=False)
    profiles.to_csv(csv_path, index=False)

    print(f"\nĐã lưu -> {pq_path}")
    print(f"Đã lưu -> {csv_path}")
    print(f"\n{'='*50}")
    print(f"Tổng số nhà hàng (Businesses) được lập cấu hình : {len(profiles):,}")
    print(f"Điểm đồ ăn (Food) trung bình      : {profiles['avg_food'].mean():.3f}")
    print(f"Điểm phục vụ (Service) trung bình : {profiles['avg_service'].mean():.3f}")
    print(f"Điểm không gian (Ambience) TB     : {profiles['avg_ambience'].mean():.3f}")
    print(f"Điểm giá trị (Price) trung bình   : {profiles['avg_price'].mean():.3f}")
    print(f"Số sao (Stars) trung bình chung   : {profiles['avg_stars'].mean():.2f}")

    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Khởi tạo bảng điểm cấu hình ABSA theo nhà hàng")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Đường dẫn Parquet file reviews (mặc định: {DATA_PATH})",
    )
    args = parser.parse_args()
    build_profiles(input_path=args.input)


if __name__ == "__main__":
    main()
