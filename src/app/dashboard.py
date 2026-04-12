"""
dashboard.py – Bảng điều khiển Phân tích Đánh giá Nhà hàng (Review Intelligence Dashboard)
                Gợi ý lai (Hybrid recommendation) tích hợp biểu đồ Radar ABSA.

Chạy:
    streamlit run src/app/dashboard.py
"""

import pickle
import re
import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# allow imports from project src/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.absa_rules import detect_aspects_in_text  # noqa: E402

# ── paths ───────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "processed" / "subset_restaurants_20000_clean.parquet"
BM25_PATH = ROOT / "models" / "bm25_index.pkl"
FAISS_PATH = ROOT / "models" / "faiss_index.bin"
DOC_IDS_PATH = ROOT / "models" / "dense_doc_ids.pkl"
ASPECT_PATH = ROOT / "evaluation" / "reports" / "aspect_profiles.parquet"

RRF_K = 60
SNIPPET_LEN = 200

SENTIMENT_COLORS = {
    "positive": "#2E7D32",  # Green
    "neutral": "#9E9E9E",   # Gray
    "negative": "#C62828",  # Red
}

# ── CSS ─────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
.stApp { background: #F7F8FA; }

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label {
    color: #6B7280 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1F2937 !important;
    font-weight: 700 !important;
}

.review-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.review-card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1F2937;
    margin: 0 0 6px 0;
}
.review-meta {
    font-size: 0.82rem;
    color: #6B7280;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.review-text {
    font-size: 0.92rem;
    color: #374151;
    line-height: 1.55;
    margin: 0;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    color: #fff;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.02em;
}
.badge-positive { background: #0F766E; }
.badge-neutral  { background: #94A3B8; }
.badge-negative { background: #DC2626; }

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1F2937;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
}

/* Custom Metric Cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    display: flex;
    align-items: center;
    gap: 16px;
}
.metric-icon {
    width: 48px;
    height: 48px;
    background: #F1F8E9;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}
.metric-content { flex: 1; }
.metric-name { font-size: 0.9rem; font-weight: 600; color: #1F2937; margin-bottom: 2px; }
.metric-value { font-size: 1.4rem; font-weight: 800; color: #1F2937; line-height: 1.1; margin-bottom: 2px; }
.metric-sub { font-size: 0.75rem; color: #6B7280; }

.hero {
    margin-bottom: 24px;
}
.hero h2 { margin: 0 0 4px; font-size: 2rem; font-weight: 800; color: #9A4A15; }
.hero h2 span { color: #1F2937; }
.hero p  { margin: 0; font-size: 1rem; color: #374151; font-weight: 500; }

.rest-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.rest-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1F2937;
    margin: 0 0 8px;
}
.rest-meta {
    font-size: 0.85rem;
    color: #6B7280;
    margin-bottom: 10px;
}
.rest-snippet {
    font-size: 0.88rem;
    color: #374151;
    font-style: italic;
    line-height: 1.5;
}
</style>
"""


# ── tải dữ liệu và mô hình (loaders) ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.warning(f"Dataset not found: {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_parquet(DATA_PATH)


@st.cache_resource(show_spinner="Loading BM25 index …")
def load_bm25():
    if not BM25_PATH.exists():
        st.warning(f"BM25 index not found: {BM25_PATH}")
        return None
    return joblib.load(BM25_PATH)


@st.cache_resource(show_spinner="Loading FAISS index …")
def load_faiss():
    if not FAISS_PATH.exists():
        st.warning(f"FAISS index not found: {FAISS_PATH}")
        return None
    return faiss.read_index(str(FAISS_PATH))


@st.cache_resource(show_spinner="Loading dense doc IDs …")
def load_doc_ids():
    if not DOC_IDS_PATH.exists():
        st.warning(f"Dense doc IDs not found: {DOC_IDS_PATH}")
        return None
    with open(DOC_IDS_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading sentence encoder …")
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Loading aspect profiles …")
def load_aspect_profiles() -> pd.DataFrame:
    if not ASPECT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(ASPECT_PATH)


# ── hàm phụ trợ (helpers) ─────────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()


def clean_query(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentiment_badge(label: str) -> str:
    return f'<span class="badge badge-{label}">{label.capitalize()}</span>'


def hybrid_search(query: str, df: pd.DataFrame, bm25, faiss_index, doc_ids, encoder, top_k: int = 100, alpha: float = 0.9) -> pd.DataFrame:
    """Lai kết quả bằng thuật toán Reciprocal Rank Fusion (wRRF) giữa BM25 + FAISS."""
    # BM25
    q_tokens = clean_query(query).split()
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_top, 1)}

    # Dense
    q_vec = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    _, faiss_top = faiss_index.search(q_vec, top_k)
    faiss_top = faiss_top[0]
    sem_ranks = {int(idx): rank for rank, idx in enumerate(faiss_top, 1) if idx >= 0}

    # RRF
    candidates = set(bm25_ranks) | set(sem_ranks)
    rows = []
    for idx in candidates:
        br = bm25_ranks.get(idx, top_k + 1)
        sr = sem_ranks.get(idx, top_k + 1)
        rrf = alpha * (1.0 / (RRF_K + br)) + (1.0 - alpha) * (1.0 / (RRF_K + sr))
        rows.append({"doc_idx": idx, "bm25_rank": br, "semantic_rank": sr, "rrf_score": rrf})

    rrf_df = pd.DataFrame(rows).sort_values("rrf_score", ascending=False).head(top_k).reset_index(drop=True)
    result = df.iloc[rrf_df["doc_idx"].tolist()].reset_index(drop=True)
    result["bm25_rank"] = rrf_df["bm25_rank"].values
    result["semantic_rank"] = rrf_df["semantic_rank"].values
    result["rrf_score"] = rrf_df["rrf_score"].values
    return result


def filter_by_vader(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    compounds = results["text"].apply(lambda t: _vader.polarity_scores(str(t))["compound"])
    res = results.copy()
    res["vader_compound"] = compounds  # Save it so aggregate_restaurants can use it
    return res[compounds >= threshold].reset_index(drop=True)


def aggregate_restaurants(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame()
    df = filtered.copy()
    df["weighted_score"] = df["rrf_score"] * (1.0 + df["vader_compound"])
    
    agg = (
        df.groupby("business_id")
        .agg(
            name=("name", "first"),
            review_count=("review_id", "count"),
            business_score=("weighted_score", "sum"),
            avg_stars=("stars", "mean"),
            positive_rate=("sentiment", lambda s: (s == "positive").mean()),
        )
        .reset_index()
    )
    agg["business_score"] = (agg["business_score"] / agg["review_count"]) * np.log10(10 + agg["review_count"])
    # We will filter by min_reviews later in the main block before head(5), or just return all and filter
    return agg.sort_values("business_score", ascending=False).reset_index(drop=True)


def make_radar(food: float, service: float, ambience: float, price: float, name: str) -> go.Figure:
    categories = ["Đồ ăn", "Phục vụ", "Không gian", "Giá cả"]
    
    # Chuyển đổi thang điểm gốc từ [-1, 1] sang thang [1, 5] sao
    def scale_score(v):
        return ((v + 1) / 2) * 4 + 1
        
    values = [scale_score(food), scale_score(service), scale_score(ambience), scale_score(price)]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(46,125,50,0.15)",
            line=dict(color="#2E7D32", width=2),
            name=name,
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[1, 5], tickmode="linear", tick0=1, dtick=1, tickfont=dict(size=9)),
        ),
        showlegend=False,
        margin=dict(t=30, b=30, l=40, r=40),
        height=260,
        paper_bgcolor="#FFFFFF",
    )
    return fig


# ── page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Review Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── load resources ──────────────────────────────────────────────────
df = load_data()
bm25 = load_bm25()
faiss_index = load_faiss()
doc_ids = load_doc_ids()
encoder = load_encoder()
aspect_profiles = load_aspect_profiles()

has_hybrid = bm25 is not None and faiss_index is not None and doc_ids is not None

# ── sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h3 style='color:#1F2937; margin-bottom:2px;'><span style='font-size:1.4rem;'>🍽️</span> Hệ thống Phân tích Đánh giá</h3>"
        "<p style='color:#6B7280; font-size:0.82rem; margin-top:0; margin-left:28px;'>"
        "Trợ lý Ẩm thực AI &amp; Tìm kiếm Khai phá</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    query = st.text_input("Tìm kiếm món ăn, trải nghiệm...", placeholder="e.g. best pizza, slow service …")
    
    with st.expander("🛠 Lọc nâng cao (Advanced Filters)"):
        alpha_val = st.slider(
            "Trọng số Hybrid wRRF (\u03B1) (Thiên về BM25)",
            min_value=0.0, max_value=1.0, value=0.9, step=0.1,
            help="=1.0: 100% BM25. =0.0: 100% FAISS (Semantic). Mặc định 0.9 để ưu tiên BM25 làm cơ sở."
        )
        
        vader_threshold = st.slider(
            "Độ tích cực tối thiểu (Min Positivity)", 
            min_value=-1.0, max_value=0.0, value=-0.3, step=0.05,
            help="Lọc bỏ các đánh giá có điểm cảm xúc VADER thấp hơn ngưỡng này."
        )
        aspect_filter = st.multiselect(
            "Lọc thế mạnh (Aspect Filter)", 
            options=["Đồ ăn", "Phục vụ", "Không gian", "Giá cả"], 
            default=[],
            help="Chỉ gợi ý các nhà hàng có điểm trung bình > 0.05 ở các khía cạnh chọn."
        )
        min_reviews = st.slider(
            "Số đánh giá tối thiểu (Min Reviews)", 
            min_value=1, max_value=20, value=1, step=1,
            help="Khắc phục Data Sparsity bằng cách ẩn nhà hàng có quá ít đánh giá trong CSDL."
        )
        top_k = st.slider(
            "Số lượng phân tích tối đa (Top-K)", 
            min_value=10, max_value=200, value=100, step=10,
            help="Giới hạn số lượng đánh giá được hệ thống truy xuất để xử lý."
        )

    show_reviews = st.toggle("Hiển thị chi tiết từng đánh giá", value=False)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.82rem; color:#6B7280;'>"
        f"<b>Corpus:</b> {len(df):,} reviews<br>"
        f"<b>Businesses:</b> {df['business_id'].nunique():,}<br>"
        f"<b>Retrieval:</b> Hybrid BM25 + FAISS (RRF)</div>",
        unsafe_allow_html=True,
    )

# ── header ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    "<h2><span style='color:#8B4513;'>Món Ăn Gì Hôm Nay?</span> 🤔</h2>"
    "<p>Nhập món ăn thêm vào thanh bên, chúng tôi sẽ tìm nhà hàng tốt nhất.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── no query → overview ────────────────────────────────────────────
def render_metric_card(icon: str, name: str, value: str, sub: str):
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-content">
            <div class="metric-name">{name}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
    </div>
    """

if not query:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(render_metric_card("💬", "Bình luận", f"{len(df):,}+", "Tổng số"), unsafe_allow_html=True)
    with c2: st.markdown(render_metric_card("🏢", "Nhà hàng", f"{df['business_id'].nunique():,}", "Trong cơ sở dữ liệu"), unsafe_allow_html=True)
    with c3: st.markdown(render_metric_card("⭐", "Điểm sao TB", f"{df['stars'].mean():.2f}", "Dựa trên bình luận"), unsafe_allow_html=True)
    pos_pct = (df["sentiment"] == "positive").mean() * 100
    with c4: st.markdown(render_metric_card("👍", "Tỷ lệ tích cực", f"{pos_pct:.1f}%", "Số cảm xúc"), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        sent_counts = df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(
            sent_counts,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title="📊 Tình cảm khách hàng",
            hole=0.6,
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent')
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1F2937"),
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.0)
        )
        # Center text inside Donut
        pos_val = sent_counts[sent_counts["sentiment"]=="positive"]["count"].values[0] / len(df) * 100
        fig_pie.add_annotation(x=0.5, y=0.5, text=f"<b>{pos_val:.1f}%</b><br>Tích cực", showarrow=False, font=dict(size=16, color="#1F2937"))
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        top_biz = df["name"].value_counts().head(10).reset_index()
        top_biz.columns = ["business", "reviews"]
        fig_bar = px.bar(
            top_biz,
            x="reviews",
            y="business",
            orientation="h",
            title="🏆 Top Nhà Hàng (theo số bình luận)",
            color_discrete_sequence=["#2E7D32"],
            text="reviews"
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed", title=""),
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1F2937"),
            margin=dict(t=40, b=10, l=10, r=40),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.stop()

# ── query entered ───────────────────────────────────────────────────
if not has_hybrid:
    st.error(
        "Hybrid search requires BM25 index, FAISS index, and dense doc IDs. "
        "Please run `build_bm25.py`, `build_dense.py` first."
    )
    st.stop()

results = hybrid_search(query, df, bm25, faiss_index, doc_ids, encoder, top_k=top_k, alpha=alpha_val)

if results.empty:
    st.warning("Không tìm thấy kết quả. Hãy thử từ khóa khác.")
    st.stop()

filtered = filter_by_vader(results, vader_threshold)
restaurants = aggregate_restaurants(filtered)

# Apply sparsity filter -> Top 5
restaurants = restaurants[restaurants["review_count"] >= min_reviews].head(5).reset_index(drop=True)

# ── KPI cards ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Tóm tắt Kết quả Tìm kiếm</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Đánh giá tìm thấy", f"{len(results):,}")
m2.metric("Sau lọc Cảm xúc", f"{len(filtered):,}")
m3.metric("Nhà hàng Gợi ý", f"{len(restaurants):,}")
m4.metric("Điểm sao TB", f"{filtered['stars'].mean():.2f}" if not filtered.empty else "–")

# ── restaurant recommendation cards ────────────────────────────────
st.markdown(f'<div class="section-title">🔍 Kết quả tìm kiếm: \'{query}\'</div>', unsafe_allow_html=True)

if restaurants.empty:
    st.info("No restaurants matched the filters.")
else:
    cols = st.columns(2)
    for i, (_, rest) in enumerate(restaurants.iterrows()):
        col = cols[i % 2]
        with col:
            biz_id = rest["business_id"]
            biz_name = rest["name"]
            avg_stars = rest["avg_stars"]
            pos_rate = rest["positive_rate"]
            review_count = rest["review_count"]

            # get top 1-2 review snippets
            biz_reviews = filtered[filtered["business_id"] == biz_id].sort_values("rrf_score", ascending=False)
            snippets = []
            if not biz_reviews.empty:
                snippets.append(str(biz_reviews.iloc[0]["text"])[:SNIPPET_LEN])
                if len(biz_reviews) > 1:
                    snippets.append(str(biz_reviews.iloc[1]["text"])[:SNIPPET_LEN])

            # aspect profile via pre-computed table
            food_s, service_s, ambience_s, price_s = 0.0, 0.0, 0.0, 0.0
            if not aspect_profiles.empty and biz_id in aspect_profiles["business_id"].values:
                profile_row = aspect_profiles[aspect_profiles["business_id"] == biz_id].iloc[0]
                food_s = profile_row.get("avg_food", 0.0)
                service_s = profile_row.get("avg_service", 0.0)
                ambience_s = profile_row.get("avg_ambience", 0.0)
                price_s = profile_row.get("avg_price", 0.0)

            # aspect filter map
            aspect_name_map = {"Đồ ăn": food_s, "Phục vụ": service_s, "Không gian": ambience_s, "Giá cả": price_s}

            if aspect_filter:
                if not all(aspect_name_map.get(a, 0) > 0.05 for a in aspect_filter):
                    continue

            stars_int = int(round(avg_stars))
            st.markdown(
                f'<div class="rest-card" style="padding: 16px; margin-bottom:0;">'
                f'<div class="rest-name">{i+1}. {biz_name}</div>'
                f'<div class="rest-meta" style="margin-bottom:0;">'
                f'<span style="color:#F59E0B;">{"⭐" * stars_int}</span> '
                f'<span style="color:#1F2937; font-weight:600;">{avg_stars:.2f}</span>'
                f'&nbsp;&middot;&nbsp; {review_count} reviews &nbsp;&middot;&nbsp; '
                f'<span style="color:#2E7D32; font-weight:600;">{pos_rate:.0%} Tích cực</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            
            # Radar smaller
            fig = make_radar(food_s, service_s, ambience_s, price_s, biz_name)
            fig.update_layout(height=200, margin=dict(t=10, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Snippet block matching the mockup st.info
            st.markdown("<b>Minh chứng ấn tượng (Top Evidence):</b>", unsafe_allow_html=True)
            for snip in snippets:
                st.info(f'"{snip}..."')
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ── review-level results (toggle) ──────────────────────────────────
if show_reviews:
    st.markdown('<div class="section-title">Chi tiết Đánh giá (Review-Level)</div>', unsafe_allow_html=True)
    st.caption(f'Hiển thị {len(filtered)} đánh giá cho truy vấn "{query}"')

    review_limit = st.slider("Giới hạn số lượng hiển thị", 10, 100, 50, 10)
    for _, row in filtered.head(review_limit).iterrows():
        stars_int = int(row["stars"])
        badge = sentiment_badge(row["sentiment"])
        rrf = row.get("rrf_score", 0)
        text = str(row["text"])[:SNIPPET_LEN]
        ellipsis = "…" if len(str(row["text"])) > SNIPPET_LEN else ""

        card = f"""
        <div class="review-card">
            <div class="review-card-title">{row.get('name', 'Unknown')}</div>
            <div class="review-meta">
                <span>{"&#9733;" * stars_int}{"&#9734;" * (5 - stars_int)}</span>
                {badge}
                <span>Độ phù hợp (Match): {rrf * 1000:.0f}</span>
            </div>
            <p class="review-text">{text}{ellipsis}</p>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)
