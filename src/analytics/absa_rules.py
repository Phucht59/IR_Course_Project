"""
absa_rules.py – Phân tích cảm xúc theo khía cạnh (ABSA) dựa trên luật (rule-based)
              sử dụng danh sách từ khóa, cửa sổ ngữ cảnh, và VADER.

Cho mỗi đánh giá, hệ thống phát hiện các khía cạnh nhà hàng (đồ ăn, phục vụ, không gian, giá cả),
trích xuất một cửa sổ ngữ cảnh ±4 từ xung quanh mỗi từ khóa, và chấm điểm bằng VADER.

Sử dụng (kiểm thử):
    python src/analytics/absa_rules.py
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# ── tập từ điển từ khóa các khía cạnh ─────────────────────────────────────────

FOOD_KEYWORDS: list[str] = [
    "food", "meal", "dish", "taste", "flavor", "flavour", "menu",
    "appetizer", "entree", "dessert", "pizza", "burger", "sushi",
    "steak", "salad", "soup", "pasta", "chicken", "seafood", "rice",
    "sauce", "cheese", "bread", "fries", "wings", "taco", "burrito",
    "sandwich", "noodle", "curry",
]

SERVICE_KEYWORDS: list[str] = [
    "service", "staff", "waiter", "waitress", "server", "manager",
    "bartender", "host", "hostess", "attentive", "rude", "friendly",
    "slow", "fast", "helpful", "polite", "unprofessional", "wait",
    "reservation", "order", "seated", "greeting", "tip", "checkout",
]

AMBIENCE_KEYWORDS: list[str] = [
    "ambience", "ambiance", "atmosphere", "decor", "decoration",
    "vibe", "music", "lighting", "noise", "noisy", "quiet", "cozy",
    "romantic", "clean", "dirty", "spacious", "crowded", "interior",
    "outdoor", "patio",
]

PRICE_KEYWORDS: list[str] = [
    "price", "prices", "expensive", "cheap", "affordable", "overpriced",
    "value", "worth", "cost", "costly", "budget", "deal", "bargain",
    "bill", "charge", "tip", "dollar", "reasonable", "pricey",
]

ASPECT_LEXICONS: dict[str, list[str]] = {
    "food": FOOD_KEYWORDS,
    "service": SERVICE_KEYWORDS,
    "ambience": AMBIENCE_KEYWORDS,
    "price": PRICE_KEYWORDS,
}

CONTEXT_WINDOW = 4  # Số lượng từ được lấy ở mỗi bên của từ khóa tìm thấy


# ── các hàm xử lý cốt lõi ──────────────────────────────────────────────────

def detect_aspects_in_text(text: str) -> dict[str, float]:
    """Trả về điểm cảm xúc của từng khía cạnh trong đoạn [-1.0, 1.0].

    Đối với mỗi khía cạnh, tìm mọi từ khóa trùng khớp trong văn bản đã tách từ,
    trích xuất một cửa sổ ngữ cảnh gồm ±4 từ xung quanh, chấm điểm vùng này bằng VADER,
    sau đó tính trung bình lại. Nếu một khía cạnh không có từ khóa nào khớp, điểm mặc định là 0.0.
    """
    tokens = text.lower().split()
    scores: dict[str, float] = {}

    for aspect, keywords in ASPECT_LEXICONS.items():
        compounds: list[float] = []
        for i, token in enumerate(tokens):
            if token in keywords:
                start = max(0, i - CONTEXT_WINDOW)
                end = min(len(tokens), i + CONTEXT_WINDOW + 1)
                window_text = " ".join(tokens[start:end])
                compound = _vader.polarity_scores(window_text)["compound"]
                compounds.append(compound)
        scores[aspect] = sum(compounds) / len(compounds) if compounds else 0.0

    return scores


def classify_aspect_sentiment(score: float) -> str:
    """Chuyển đổi điểm số liên tục thành nhãn phân loại rời rạc."""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"


def get_aspect_labels(scores: dict[str, float]) -> dict[str, str]:
    """Chuyển đổi mảng điểm số khía cạnh dạng số sang nhãn cảm xúc dạng chữ."""
    return {aspect: classify_aspect_sentiment(s) for aspect, s in scores.items()}


# ── hàm kiểm thử cho CLI ────────────────────────────────────────────────────────

def main() -> None:
    examples = [
        "The food was absolutely delicious but the waiter was very rude and slow.",
        "Great atmosphere and reasonable prices, but the pasta was bland.",
        "Terrible service, overpriced menu, noisy atmosphere. Never coming back.",
    ]
    for text in examples:
        scores = detect_aspects_in_text(text)
        labels = get_aspect_labels(scores)
        print(f"\nVăn bản: {text[:80]}…")
        for asp in ("food", "service", "ambience", "price"):
            print(f"  {asp:10s}  điểm={scores[asp]:+.3f}  -> {labels[asp]}")


if __name__ == "__main__":
    main()
