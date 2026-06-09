# -*- coding: utf-8 -*-
"""
AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam
=============================================================
Phiên bản giao diện MỚI (theme sáng, điều hướng dạng tab theo cấp độ).
Toàn bộ NỘI DUNG TÍNH TOÁN giữ nguyên so với bản trước.

Bài tập lớn: Các mô hình ra quyết định
Họ và tên : Trần Dương Nhi
Mã sinh viên: ____________

Chạy:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="AIDEOM-VN · Mô hình ra quyết định",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# PALETTE (theme sáng — indigo / teal)
# ------------------------------------------------------------
ACC1 = "#6366f1"   # indigo (primary)
ACC2 = "#0ea5e9"   # sky / teal (secondary)
ACC3 = "#f59e0b"   # amber (tertiary)
EDGE = "#ffffff"   # viền trắng cho bar trên nền sáng

# ------------------------------------------------------------
# CSS — nền sáng, thẻ mềm, header gradient
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
        --c1:#6366f1; --c2:#0ea5e9; --c3:#f59e0b;
        --card:#ffffff; --soft:#f8fafc;
    }
    .stApp { background: #f1f5f9; }
    .block-container { padding-top: 1.2rem; max-width: 1280px; }
    h1,h2,h3,h4 { color: var(--ink); }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg,#4f46e5 0%,#6366f1 45%,#0ea5e9 100%);
        border-radius: 20px; padding: 30px 34px; color:#fff;
        box-shadow: 0 12px 30px rgba(79,70,229,.25); margin-bottom: 20px;
    }
    .hero h1 { color:#fff; font-size:2.5rem; font-weight:800; margin:0; letter-spacing:-1px;}
    .hero .tag { font-size:1.05rem; opacity:.92; font-weight:500; }
    .hero .flag { font-size:1.1rem; opacity:.85; margin-top:6px;}

    /* KPI chips */
    .kpi {
        background: var(--card); border-radius:16px; padding:16px 18px;
        border:1px solid var(--line); box-shadow:0 4px 14px rgba(2,6,23,.05);
    }
    .kpi .lab { color:var(--muted); font-size:.85rem; font-weight:600; }
    .kpi .val { color:var(--ink); font-size:1.7rem; font-weight:800; line-height:1.15;}
    .kpi .dl  { display:inline-block; margin-top:6px; background:#ecfdf5; color:#059669;
                padding:1px 9px; border-radius:14px; font-size:.78rem; font-weight:700;}

    /* Card chung */
    .softcard {
        background:var(--card); border-radius:16px; padding:18px 20px;
        border:1px solid var(--line); box-shadow:0 4px 14px rgba(2,6,23,.05);
    }
    .pill { display:inline-block;padding:3px 12px;border-radius:999px;
            font-weight:700;font-size:.8rem; }

    /* Tabs lớn hơn */
    .stTabs [data-baseweb="tab-list"]{ gap:4px; flex-wrap:wrap; }
    .stTabs [data-baseweb="tab"]{
        background:#fff;border:1px solid var(--line);border-bottom:none;
        border-radius:12px 12px 0 0;padding:9px 14px;font-weight:600;color:var(--ink);
    }
    .stTabs [aria-selected="true"]{ background:var(--c1);color:#fff;border-color:var(--c1);}

    /* selectbox điều hướng */
    div[data-testid="stSelectbox"] label { font-weight:700;color:var(--ink); }

    .sig { background:linear-gradient(135deg,#eef2ff,#e0f2fe); border-radius:14px;
           padding:16px 18px; border:1px solid #c7d2fe; }
    .sig .nm { font-weight:800;color:#3730a3;font-size:1.05rem; }
    .sig .mt { color:#475569;font-size:.88rem; }
    div[data-testid="stMetricValue"]{ color:var(--c1); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Matplotlib — theme SÁNG
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "text.color": "#0f172a",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "axes.edgecolor": "#cbd5e1",
    "grid.color": "#e2e8f0",
    "font.size": 10,
})

# ============================================================
# NẠP DỮ LIỆU
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _find(fname):
    for d in (DATA_DIR, os.path.join(DATA_DIR, "data"), "."):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return fname


@st.cache_data(show_spinner=False)
def load_macro():
    df = pd.read_csv(_find("vietnam_macro_2020_2025.csv"))
    return df.sort_values("year").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sectors():
    return pd.read_csv(_find("vietnam_sectors_2024.csv"))


@st.cache_data(show_spinner=False)
def load_regions():
    return pd.read_csv(_find("vietnam_regions_2024.csv"))


SECTOR_VI = [
    "Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Khai khoáng",
    "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải",
    "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế",
]
REGION_VI = [
    "Trung du miền núi phía Bắc", "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ", "Tây Nguyên", "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]
REGION_CODE = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
EN2CODE = {
    "Northern Midlands and Mountains": "NMM",
    "Red River Delta": "RRD",
    "North Central and South Central Coast": "NCC",
    "Central Highlands": "CH",
    "Southeast": "SE",
    "Mekong Delta": "MD",
}

# ============================================================
# HÀM DÙNG CHUNG
# ============================================================
ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {"I": "Hạ tầng số", "D": "Chuyển đổi số DN", "AI": "Năng lực AI", "H": "Nhân lực số"}

# Hệ số tác động biên β[vùng, hạng mục] (Bài 4 / 7 / 12)
BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}


def topsis(X, w, is_benefit):
    """TOPSIS chuẩn hóa vector — trả về hệ số gần gũi C*."""
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * w
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    return S_neg / (S_star + S_neg)


def entropy_weights(X):
    P = X / X.sum(axis=0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()



# ============================================================
# OVERRIDE section() — kiểu tiêu đề mới (gạch màu trái)
# ============================================================
def section(title, desc=None):
    st.markdown(
        f'<div style="border-left:4px solid {ACC1};padding-left:12px;margin:18px 0 6px;">'
        f'<span style="font-weight:800;font-size:1.12rem;color:#0f172a;">{title}</span></div>',
        unsafe_allow_html=True,
    )
    if desc:
        st.caption(desc)


# ============================================================
# TRANG CHỦ (giao diện mới — dashboard)
# ============================================================
def page_home():
    st.markdown(
        """
        <div class="hero">
          <h1>🧭 AIDEOM-VN</h1>
          <div class="tag"><b>AI-Driven Decision Optimization Model for Vietnam</b></div>
          <div class="flag">12 bài toán mô hình ra quyết định · dữ liệu thực tế Việt Nam 2020–2025</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        ("GDP 2025", "514,0 tỷ USD", "▲ 8,02%"),
        ("Kinh tế số / GDP", "≈ 19,5%", "▲ 1,2 đpt"),
        ("FDI giải ngân 2025", "27,6 tỷ USD", "▲ 8,9%"),
        ("GDP/người 2025", "5.026 USD", "▲ 6,9%"),
    ]
    for col, (lab, val, dl) in zip(st.columns(4), cards):
        col.markdown(
            f'<div class="kpi"><div class="lab">{lab}</div>'
            f'<div class="val">{val}</div><div class="dl">{dl}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### 📚 Lộ trình 12 bài theo 4 cấp độ")
        levels = [
            ("Dễ", "#dcfce7", "#16a34a", [
                ("1", "Cobb-Douglas mở rộng + AI — growth accounting, dự báo GDP 2030"),
                ("2", "LP phân bổ ngân sách 4 hạng mục — shadow price"),
                ("3", "Chỉ số ưu tiên 10 ngành — min-max, sensitivity"),
            ]),
            ("Trung bình", "#fef9c3", "#ca8a04", [
                ("4", "LP ngân sách ngành-vùng — PuLP + công bằng vùng"),
                ("5", "MIP chọn 15 dự án — knapsack, precedence"),
                ("6", "TOPSIS 6 vùng — entropy, AHP"),
            ]),
            ("Khá khó", "#ffedd5", "#ea580c", [
                ("7", "NSGA-II Pareto — 4 mục tiêu xung đột"),
                ("8", "Tối ưu động 2026-2035 — CRRA, SLSQP"),
                ("9", "AI & lao động — NetJob, đào tạo lại"),
            ]),
            ("Khó", "#fee2e2", "#dc2626", [
                ("10", "Stochastic 2 giai đoạn — VSS, EVPI"),
                ("11", "Q-learning chính sách thích nghi"),
                ("12", "AIDEOM tích hợp 6 module · dashboard"),
            ]),
        ]
        for name, bg, fg, rows in levels:
            with st.container():
                st.markdown(
                    f'<span class="pill" style="background:{bg};color:{fg};">{name}</span>',
                    unsafe_allow_html=True,
                )
                for code, desc in rows:
                    st.markdown(
                        f'<div style="display:flex;gap:10px;padding:5px 0;">'
                        f'<div style="min-width:30px;font-weight:800;color:{ACC1};">B{code}</div>'
                        f'<div style="color:#334155;">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )

    with right:
        st.markdown("#### 🧑‍🎓 Thông tin")
        st.markdown(
            """
            <div class="sig">
              <div class="nm">Trần Dương Nhi</div>
              <div class="mt">Mã sinh viên: ____________</div>
              <div class="mt">Bài tập lớn: Các mô hình ra quyết định</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 🛠️ Công cụ")
        st.markdown(
            '<div class="softcard" style="color:#334155;">'
            "NumPy · pandas · SciPy · PuLP · CVXPY · Pyomo · pymoo · Gymnasium · Streamlit"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🗃️ Dữ liệu Việt Nam 2020–2025")
    t1, t2, t3, t4 = st.tabs(["📈 Vĩ mô", "🏭 10 ngành", "🗺️ 6 vùng", "📋 Tham chiếu"])
    with t1:
        st.dataframe(load_macro(), width='stretch', hide_index=True)
    with t2:
        sec = load_sectors().copy(); sec.insert(2, "Tên VN", SECTOR_VI)
        st.dataframe(sec, width='stretch', hide_index=True)
    with t3:
        reg = load_regions().copy(); reg.insert(2, "Tên VN", REGION_VI)
        st.dataframe(reg, width='stretch', hide_index=True)
    with t4:
        st.dataframe(pd.DataFrame({
            "Chỉ tiêu": ["GDP (nghìn tỷ VND)", "GDP (tỷ USD)", "GDP/người (USD)",
                          "Tăng trưởng GDP (%)", "Dân số (triệu)", "FDI giải ngân (tỷ USD)",
                          "Xuất khẩu (tỷ USD)", "Kinh tế số/GDP (%)", "DN công nghệ số",
                          "Startup AI", "GII (/139)"],
            "2024": ["11.511,9", "476,3", "4.700", "7,09", "101,3", "25,35", "405,5",
                      "18,3", "73.788", "≈278", "44"],
            "2025": ["12.847,6", "514,0", "5.026", "8,02", "102,3", "27,60", "475,0",
                      "≈19,5", "80.052", "≈350+", "44"],
        }), width='stretch', hide_index=True)

# ============================================================
# BÀI 1 — COBB-DOUGLAS MỞ RỘNG
# ============================================================
def page_bai1():
    st.markdown("## 🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng (AI & số hóa)")
    st.caption("Growth accounting trên dữ liệu VN 2020-2025 · dự báo GDP 2030")

    df = load_macro()
    years = df["year"].values
    Y = df["GDP_trillion_VND"].values
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900], float)
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = df["digital_economy_share_GDP_pct"].values.astype(float)
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

    with st.expander("⚙️ Hệ số co giãn (có thể tinh chỉnh)", expanded=True):
        c = st.columns(5)
        alpha = c[0].number_input("α (vốn K)", 0.0, 1.0, 0.33, 0.01)
        beta = c[1].number_input("β (lao động L)", 0.0, 1.0, 0.42, 0.01)
        gamma = c[2].number_input("γ (số hóa D)", 0.0, 1.0, 0.10, 0.01)
        delta = c[3].number_input("δ (AI)", 0.0, 1.0, 0.08, 0.01)
        theta = c[4].number_input("θ (nhân lực H)", 0.0, 1.0, 0.07, 0.01)
        s = alpha + beta + gamma + delta + theta
        st.caption(f"Tổng hệ số = **{s:.2f}** "
                   f"{'✅ lợi suất không đổi theo quy mô' if abs(s-1) < 1e-9 else '⚠️ ≠ 1'}")

    A = Y / (K ** alpha * L ** beta * D ** gamma * AI ** delta * H ** theta)

    section("Câu 1.4.1 — Ước lượng TFP A_t (giải ngược hàm sản xuất)")
    c1, c2 = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, A, "o-", color=ACC1, lw=2, ms=8)
    for x, a in zip(years, A):
        ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("Năm"); ax.set_ylabel("A_t (TFP)")
    ax.set_title("Năng suất nhân tố tổng hợp 2020-2025"); ax.grid(alpha=.3)
    c1.pyplot(fig)
    c2.dataframe(pd.DataFrame({"Năm": years, "Y thực tế": Y.round(1),
                              "A_t (TFP)": A.round(4)}),
                 width='stretch', hide_index=True)
    g_A_tb = ((A[-1] / A[0]) ** (1 / 5) - 1) * 100
    c2.metric("TFP tăng bình quân", f"{g_A_tb:.2f}%/năm")

    section("Câu 1.4.2 — Dự báo Ŷ với A trung bình & MAPE")
    A_mean = A.mean()
    Y_hat = A_mean * (K ** alpha * L ** beta * D ** gamma * AI ** delta * H ** theta)
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    c1, c2 = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, Y, "o-", color=ACC2, lw=2, label="Y thực tế")
    ax.plot(years, Y_hat, "s--", color=ACC1, lw=2, label="Ŷ dự báo")
    ax.set_xlabel("Năm"); ax.set_ylabel("GDP (nghìn tỷ VND)")
    ax.legend(); ax.grid(alpha=.3); ax.set_title("Dự báo vs Thực tế")
    c1.pyplot(fig)
    c2.metric("MAPE", f"{mape:.3f}%")
    c2.dataframe(pd.DataFrame({"Năm": years, "Y": Y.round(0),
                              "Ŷ": Y_hat.round(0),
                              "Sai số %": ((Y_hat - Y) / Y * 100).round(2)}),
                 width='stretch', hide_index=True)

    section("Câu 1.4.3 — Phân rã tăng trưởng 2020-2025")
    n = 5
    g = {k: (np.log(v[-1]) - np.log(v[0])) / n
         for k, v in {"Y": Y, "K": K, "L": L, "D": D, "AI": AI, "H": H, "A": A}.items()}
    contrib = {
        "TFP": g["A"], "K (Vốn)": alpha * g["K"], "L (Lao động)": beta * g["L"],
        "D (Số hóa)": gamma * g["D"], "AI": delta * g["AI"], "H (Nhân lực)": theta * g["H"],
    }
    pct = {k: v / g["Y"] * 100 for k, v in contrib.items()}
    c1, c2 = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [ACC2, ACC3, "#ef4444", "#f59e0b", "#a855f7", "#06b6d4"]
    bars = ax.bar(list(pct.keys()), list(pct.values()), color=cols, edgecolor=EDGE)
    for b, v in zip(bars, pct.values()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .5,
                f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Đóng góp (%)"); ax.set_title("Phân rã tăng trưởng GDP bình quân năm")
    ax.axhline(0, color="#3a4257"); ax.grid(axis="y", alpha=.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    c1.pyplot(fig)
    c2.dataframe(pd.DataFrame({
        "Yếu tố": list(contrib.keys()),
        "Đóng góp (%)": [round(v * 100, 3) for v in contrib.values()],
        "Tỷ lệ (%)": [round(v, 2) for v in pct.values()],
    }), width='stretch', hide_index=True)
    c2.metric("Tăng trưởng GDP bình quân", f"{g['Y']*100:.2f}%/năm")

    section("Câu 1.4.4 — Kịch bản dự báo GDP 2030")
    c = st.columns(4)
    D30 = c[0].slider("D 2030 (% GDP)", 20.0, 40.0, 30.0, .5)
    AI30 = c[1].slider("AI 2030 (nghìn DN)", 80, 150, 100, 5)
    H30 = c[2].slider("H 2030 (%)", 30.0, 45.0, 35.0, .5)
    tfp_g = c[3].slider("TFP tăng (%/năm)", 0.0, 3.0, 1.2, .1)
    K30 = K[-1] * 1.06 ** 5
    L30 = L[-1] * 1.005 ** 5
    A30 = A[-1] * (1 + tfp_g / 100) ** 5
    Y30 = A30 * (K30 ** alpha * L30 ** beta * D30 ** gamma * AI30 ** delta * H30 ** theta)
    gr = ((Y30 / Y[-1]) ** (1 / 5) - 1) * 100
    usd = Y30 * 1e12 / (110e6) / 25500
    c = st.columns(3)
    c[0].metric("GDP 2030 dự báo", f"{Y30:,.0f} nghìn tỷ")
    c[1].metric("Tăng trưởng 2025-2030", f"{gr:.2f}%/năm")
    c[2].metric("GDP/người 2030", f"≈ {usd:,.0f} USD")

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- **TFP tăng** trong giai đoạn 2020-2025 phản ánh chất lượng tăng trưởng "
            "cải thiện — nền kinh tế dựa nhiều hơn vào năng suất, không chỉ vào tích lũy vốn.\n"
            "- Trong ba yếu tố mới (D, AI, H), **số hóa D** thường đóng góp lớn nhất do tốc độ "
            "tăng nhanh (12% → 19,5% GDP).\n"
            "- Mục tiêu **30% kinh tế số/GDP vào 2030** khả thi nếu duy trì đà tăng D và TFP, "
            "nhưng cần ràng buộc về đầu tư hạ tầng số và nhân lực số đi kèm."
        )


# ============================================================
# BÀI 2 — LP NGÂN SÁCH 4 HẠNG MỤC
# ============================================================
def page_bai2():
    from scipy.optimize import linprog
    st.markdown("## 💰 Bài 2 — LP phân bổ ngân sách 4 hạng mục đầu tư số")
    st.caption("scipy.optimize.linprog · giá đối ngẫu (shadow price) · phân tích độ nhạy")

    st.latex(r"\max Z = 0{,}85x_1 + 1{,}20x_2 + 0{,}95x_3 + 1{,}35x_4")
    st.caption("x₁ hạ tầng số · x₂ AI & dữ liệu · x₃ nhân lực số · x₄ R&D công nghệ (nghìn tỷ VND)")

    c = st.columns(2)
    B = c[0].slider("Ngân sách tổng (nghìn tỷ VND)", 100, 200, 100, 10)
    x3min = c[1].slider("Sàn nhân lực số x₃ ≥", 20, 40, 20, 1)

    cobj = [-0.85, -1.20, -0.95, -1.35]
    A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0],
            [0, 0, -1, 0], [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]]
    b_ub = [B, -25, -15, -x3min, -10, 0]
    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")

    section("Câu 2.4.1 / 2.4.4 — Nghiệm tối ưu")
    if res.success:
        names = ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D"]
        c1, c2 = st.columns([2, 3])
        c1.metric("Z* (GDP tăng thêm)", f"{-res.fun:.2f} nghìn tỷ")
        c1.metric("Tỷ trọng AI+R&D", f"{(res.x[1]+res.x[3])/res.x.sum()*100:.1f}%")
        fig, ax = plt.subplots(figsize=(7, 4))
        cols = [ACC3, "#a855f7", "#f59e0b", ACC1]
        ax.bar(names, res.x, color=cols, edgecolor=EDGE)
        for i, v in enumerate(res.x):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
        ax.set_ylabel("Nghìn tỷ VND"); ax.grid(axis="y", alpha=.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        c2.pyplot(fig)
    else:
        st.error("Bài toán KHÔNG khả thi với cấu hình hiện tại.")

    section("Câu 2.4.3 — Phân tích độ nhạy: đường cong Z*(B)")
    Bs = np.arange(100, 201, 10)
    Zs = []
    for b in Bs:
        r = linprog(cobj, A_ub=[[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0],
                                [0, 0, -1, 0], [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]],
                    b_ub=[b, -25, -15, -20, -10, 0], bounds=[(0, None)] * 4, method="highs")
        Zs.append(-r.fun if r.success else np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Bs, Zs, "o-", color=ACC1, lw=2)
    ax.set_xlabel("Ngân sách tổng B (nghìn tỷ)"); ax.set_ylabel("Z* (GDP gain)")
    ax.set_title("Đường cong Z*(B) — slope = shadow price ngân sách"); ax.grid(alpha=.3)
    st.pyplot(fig)
    st.info("**Shadow price ngân sách = 1,35**: mỗi nghìn tỷ ngân sách tăng thêm tạo "
            "thêm 1,35 nghìn tỷ GDP — bằng hệ số R&D (cao nhất), vì phần tăng được dồn vào R&D.")

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Shadow price ngân sách (1,35) là **cận trên hợp lý của chi phí cơ hội vốn công**: "
            "chừng nào lợi suất biên còn > chi phí huy động vốn thì nên mở rộng ngân sách.\n"
            "- R&D có hệ số tác động cao nhất nhưng sàn tối thiểu thấp nhất do **rủi ro & độ trễ "
            "tác động dài hạn** khiến nhà hoạch định thận trọng.\n"
            "- Tỷ lệ 35% công nghệ chiến lược (AI+R&D) là tham vọng so với cơ cấu ngân sách "
            "thực tế đang ưu tiên hạ tầng giao thông và an sinh xã hội."
        )


# ============================================================
# BÀI 3 — CHỈ SỐ ƯU TIÊN NGÀNH
# ============================================================
def page_bai3():
    st.markdown("## 📊 Bài 3 — Chỉ số ưu tiên ngành cho 10 ngành Việt Nam")
    st.caption("Chuẩn hóa min-max · weighted scoring · phân tích độ nhạy trọng số")

    df = load_sectors().copy()
    df["sector_vi"] = SECTOR_VI
    GDP24 = 11511.9
    df["productivity"] = (df["gdp_share_2024_pct"] / 100) * GDP24 / df["labor_million"]

    cols_good = ["growth_rate_2024_pct", "productivity", "spillover_coef_0_1",
                 "export_billion_USD", "labor_million", "ai_readiness_0_100"]
    col_bad = "automation_risk_pct"

    def ng(x): return (x - x.min()) / (x.max() - x.min())
    def nb(x): return (x.max() - x) / (x.max() - x.min())

    Xg = df[cols_good].apply(ng)
    Xb = nb(df[col_bad])

    section("Câu 3.4.2 — Chỉ số Priority với bộ trọng số (điều chỉnh được)")
    st.caption("Trọng số chuẩn hóa lại để tổng = 1.")
    labels = ["Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI readiness", "Rủi ro TĐH"]
    defaults = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15]
    c = st.columns(7)
    raw = [c[i].number_input(labels[i], 0.0, 1.0, defaults[i], 0.01, key=f"w3_{i}") for i in range(7)]
    tot = sum(raw)
    w = np.array(raw[:6]) / tot
    w_risk = raw[6] / tot

    priority = Xg.values @ w + w_risk * Xb.values
    df["Priority"] = priority
    rank = df[["sector_vi", "Priority"]].sort_values("Priority", ascending=False).reset_index(drop=True)
    rank.index += 1

    c1, c2 = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 5))
    order = df.sort_values("Priority")
    cc = [ACC1 if i >= 7 else ACC3 for i in range(10)]
    ax.barh(order["sector_vi"], order["Priority"], color=cc, edgecolor=EDGE)
    ax.set_xlabel("Priority"); ax.set_title("Xếp hạng ưu tiên 10 ngành"); ax.grid(axis="x", alpha=.3)
    c1.pyplot(fig)
    rank2 = rank.copy(); rank2["Priority"] = rank2["Priority"].round(4)
    rank2.columns = ["Ngành", "Priority"]
    c2.dataframe(rank2, width='stretch')
    c2.success(f"🥇 Top-3: {', '.join(rank['sector_vi'].head(3))}")

    section("Câu 3.4.3 — Độ nhạy theo w_AI (heatmap)")
    w_ai_range = np.arange(0.05, 0.45, 0.05)
    base = np.array([0.15, 0.15, 0.20, 0.15, 0.10])
    hm = []
    for wai in w_ai_range:
        rem = 1.0 - wai - 0.15
        ws = base * (rem / base.sum())
        wf = np.append(ws, wai)
        hm.append(Xg.values @ wf + 0.15 * Xb.values)
    hm = np.array(hm)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(w_ai_range))); ax.set_yticklabels([f"{w:.2f}" for w in w_ai_range])
    ax.set_xticks(range(10)); ax.set_xticklabels([s[:10] for s in SECTOR_VI], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("w_AI"); ax.set_title("Priority theo trọng số AI Readiness")
    plt.colorbar(im, label="Priority"); fig.tight_layout()
    st.pyplot(fig)

    section("Câu 3.4.4 — So sánh 2 định hướng chính sách")
    w_growth = np.array([0.25, 0.25, 0.10, 0.25, 0.05, 0.05]); wg_risk = 0.05
    w_incl = np.array([0.05, 0.10, 0.25, 0.05, 0.25, 0.10]); wi_risk = 0.20
    pg = Xg.values @ w_growth + wg_risk * Xb.values
    pi = Xg.values @ w_incl + wi_risk * Xb.values
    rg = pd.Series(pg, index=SECTOR_VI).sort_values(ascending=False)
    ri = pd.Series(pi, index=SECTOR_VI).sort_values(ascending=False)
    c1, c2 = st.columns(2)
    c1.markdown("**🚀 Định hướng tăng trưởng**")
    c1.success("Top-3: " + ", ".join(rg.head(3).index))
    c1.dataframe(rg.round(4).rename("Priority"), width='stretch')
    c2.markdown("**🤝 Định hướng bao trùm**")
    c2.success("Top-3: " + ", ".join(ri.head(3).index))
    c2.dataframe(ri.round(4).rename("Priority"), width='stretch')

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Ngành **Khai khoáng** có năng suất rất cao nhưng tăng trưởng âm, lan tỏa thấp "
            "và rủi ro tự động hóa lớn → không lọt nhóm ưu tiên dù năng suất cao.\n"
            "- Kết quả Top-3 thường gồm các ngành lan tỏa & AI readiness cao (CNTT-TT, "
            "Tài chính-Ngân hàng) — phù hợp tinh thần Nghị quyết 57-NQ/TW.\n"
            "- Trọng số nên do **quy trình đối thoại công khai** giữa chuyên gia kỹ thuật và "
            "hội đồng chính sách quyết định để bảo đảm tính chính danh."
        )


# ============================================================
# BÀI 4 — LP NGÀNH-VÙNG
# ============================================================
@st.cache_data(show_spinner=False)
def _solve_lp4(B, with_equity, lam):
    import pulp
    gamma_val = 0.002
    reg = load_regions()
    D0 = dict(zip(reg["region_name_en"].map(EN2CODE), reg["digital_index_0_100"]))
    m = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGION_CODE, ITEMS), lowBound=0)
    m += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGION_CODE for j in ITEMS)
    m += pulp.lpSum(x[r][j] for r in REGION_CODE for j in ITEMS) <= B
    for r in REGION_CODE:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= 5000
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= 12000
    m += pulp.lpSum(x[r]["H"] for r in REGION_CODE) >= 12000
    if with_equity:
        M = pulp.LpVariable("Dmax")
        for r in REGION_CODE:
            m += D0[r] + gamma_val * x[r]["D"] <= M
            m += D0[r] + gamma_val * x[r]["D"] >= lam * M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    mat = np.array([[x[r][j].value() for j in ITEMS] for r in REGION_CODE])
    return mat, float(pulp.value(m.objective))


def page_bai4():
    st.markdown("## 🗺️ Bài 4 — LP phân bổ ngân sách số theo ngành-vùng")
    st.caption("PuLP (CBC) · 24 biến · ràng buộc công bằng vùng miền (Mục 7.3)")

    c = st.columns(2)
    B = c[0].slider("Ngân sách tổng (tỷ VND)", 40000, 60000, 50000, 2000)
    lam = c[1].slider("Hệ số công bằng λ", 0.5, 0.9, 0.6, 0.05)

    mat, Z = _solve_lp4(B, True, lam)
    mat_ne, Z_ne = _solve_lp4(B, False, lam)

    section("Câu 4.4.1 / 4.4.3 — Phân bổ tối ưu & heatmap")
    c1, c2 = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(6)); ax.set_yticklabels([r[:18] for r in REGION_VI], fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(ITEMS)
    ax.set_title(f"Phân bổ tối ưu (Z*={Z:,.0f} tỷ)")
    for i in range(6):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 7000 else "black")
    plt.colorbar(im, label="tỷ VND"); fig.tight_layout()
    c1.pyplot(fig)
    dfm = pd.DataFrame(mat, columns=ITEMS, index=REGION_VI).round(0)
    dfm["Tổng"] = dfm.sum(axis=1)
    c2.dataframe(dfm, width='stretch')
    c2.metric("Z* (GDP gain)", f"{Z:,.0f} tỷ VND")

    section("Câu 4.4.4 — Chi phí kinh tế của công bằng vùng miền")
    c = st.columns(3)
    c[0].metric("Z* CÓ công bằng", f"{Z:,.0f}")
    c[1].metric("Z* KHÔNG công bằng", f"{Z_ne:,.0f}")
    c[2].metric("Chi phí công bằng", f"{Z_ne - Z:,.0f} tỷ",
                f"-{(Z_ne-Z)/Z_ne*100:.2f}%")

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Nếu bỏ ràng buộc công bằng, vốn dồn về **ĐB sông Hồng & Đông Nam Bộ** "
            "(hệ số AI cao nhất 1,40–1,55) → đào sâu chênh lệch vùng.\n"
            "- Trần ngân sách mỗi vùng đóng vai trò **chính sách phân quyền**, giảm Z* một phần "
            "nhưng đổi lấy cân bằng phát triển.\n"
            "- Tây Nguyên có hệ số AI thấp (0,45) nhưng hệ số H cao (1,35) → mô hình ưu tiên "
            "đầu tư **nhân lực số (H) và hạ tầng (I)** trước khi đẩy mạnh AI."
        )


# ============================================================
# BÀI 5 — MIP CHỌN DỰ ÁN
# ============================================================
PROJ_NAMES = {
    1: "TT dữ liệu Hòa Lạc", 2: "TT dữ liệu phía Nam", 3: "5G toàn quốc",
    4: "VNeID 2.0", 5: "Cổng DVC v3", 6: "Y tế số", 7: "Giáo dục số K-12",
    8: "TT AI + supercomputing", 9: "Fintech sandbox", 10: "Logistics thông minh",
    11: "Nông nghiệp số ĐBSCL", 12: "Đào tạo 50K kỹ sư AI", 13: "Khu CN bán dẫn BN-BG",
    14: "An ninh mạng SOC", 15: "Open Data",
}


@st.cache_data(show_spinner=False)
def _solve_mip(budget_total, budget_12, use_expected, force_p1p2):
    from pulp import (LpProblem, LpMaximize, LpVariable, lpSum, LpStatus,
                      PULP_CBC_CMD, value)
    P = list(range(1, 16))
    C = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500, 8: 15000,
         9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
    C1 = {1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500, 6: 4000, 7: 4500, 8: 9000,
          9: 1800, 10: 5000, 11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200}
    B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
         9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
    fields = {1: "ht", 2: "ht", 3: "ht", 4: "cp", 5: "cp", 6: "yt", 7: "gd", 8: "ai",
              9: "tc", 10: "lg", 11: "nn", 12: "nl", 13: "bd", 14: "an", 15: "dl"}
    prob = {"ht": .85, "cp": .75, "ai": .65, "bd": .65, "yt": .8, "gd": .8, "tc": .8,
            "lg": .8, "nn": .8, "nl": .8, "an": .8, "dl": .8}
    m = LpProblem("VN_Project_Selection", LpMaximize)
    y = LpVariable.dicts("y", P, cat="Binary")
    if use_expected:
        m += lpSum(prob[fields[i]] * B[i] * y[i] for i in P)
    else:
        m += lpSum(B[i] * y[i] for i in P)
    m += lpSum(C[i] * y[i] for i in P) <= budget_total
    m += lpSum(C1[i] * y[i] for i in P) <= budget_12
    if not force_p1p2:
        m += y[1] + y[2] <= 1
    else:
        m += y[1] >= 1; m += y[2] >= 1
    m += y[8] <= y[12]; m += y[13] <= y[12]
    m += y[4] + y[5] >= 1; m += y[14] >= 1
    m += lpSum(y[i] for i in P) >= 7; m += lpSum(y[i] for i in P) <= 11
    m.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[m.status]
    sel = [i for i in P if y[i].value() and y[i].value() > 0.5]
    Z = value(m.objective)
    rows = [{"Mã": f"P{i}", "Tên dự án": PROJ_NAMES[i], "Chi phí": C[i],
             "NPV": B[i], "NPV/C": round(B[i] / C[i], 2)} for i in sel]
    return status, sel, Z, sum(C[i] for i in sel), rows


def page_bai5():
    st.markdown("## 🎯 Bài 5 — MIP lựa chọn dự án chuyển đổi số (15 dự án)")
    st.caption("PuLP + CBC · knapsack tổng quát · ràng buộc loại trừ, tiên quyết, ngân sách đa năm")

    c = st.columns(3)
    bt = c[0].slider("Ngân sách 5 năm (tỷ)", 80000, 120000, 80000, 5000)
    use_exp = c[1].toggle("Tối đa hóa lợi ích kỳ vọng (rủi ro)", value=False)
    force = c[2].toggle("Bắt buộc cả P1 & P2 (redundancy)", value=False)

    status, sel, Z, cost, rows = _solve_mip(bt, 40000, use_exp, force)
    section("Câu 5.4.1–5.4.4 — Tập dự án tối ưu")
    if status == "Optimal":
        c = st.columns(3)
        c[0].metric("Số dự án chọn", len(sel))
        c[1].metric("Tổng chi phí", f"{cost:,} tỷ")
        c[2].metric("Tổng lợi ích Z*", f"{Z:,.0f} tỷ")
        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch', hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(df["Mã"], df["NPV"], color=ACC2, label="NPV", edgecolor=EDGE)
        ax.bar(df["Mã"], df["Chi phí"], color=ACC1, alpha=.7, label="Chi phí")
        ax.set_ylabel("tỷ VND"); ax.legend(); ax.grid(axis="y", alpha=.3)
        ax.set_title("NPV vs Chi phí các dự án được chọn")
        st.pyplot(fig)
    else:
        st.error(f"Bài toán {status} — không tìm được nghiệm khả thi.")

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- P15 (Open Data) có **tỷ suất NPV/chi phí cao nhất (≈2,53)** nên thực tế "
            "thường ĐƯỢC chọn — giả thiết 'mô hình bỏ qua P15' trong đề bài cần kiểm chứng lại.\n"
            "- Ràng buộc **bắt buộc P14 (an ninh mạng)** có thể làm giảm Z* một chút nhưng hợp lý "
            "vì an ninh dữ liệu là điều kiện nền của chuyển đổi số.\n"
            "- Hiệu ứng cộng hưởng giữa P8 (AI) và P13 (bán dẫn) có thể mô hình hóa bằng biến "
            "tích y₈·y₁₃ tuyến tính hóa, thêm lợi ích nếu chọn cả hai."
        )


# ============================================================
# BÀI 6 — TOPSIS 6 VÙNG
# ============================================================
def page_bai6():
    st.markdown("## 🏆 Bài 6 — TOPSIS xếp hạng 6 vùng theo ưu tiên đầu tư AI")
    st.caption("Chuẩn hóa vector · trọng số chuyên gia vs Entropy · phân tích độ nhạy")

    df = load_regions().copy()
    criteria = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD",
                "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct",
                "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
    clabels = ["GRDP/N", "FDI", "Digital", "AI", "LĐĐT", "R&D", "Internet", "Gini"]
    is_ben = np.array([True, True, True, True, True, True, True, False])
    X = df[criteria].values.astype(float)

    section("Câu 6.4.1 — Trọng số chuyên gia (điều chỉnh được)")
    defaults = [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10]
    cc = st.columns(8)
    raw = [cc[i].number_input(clabels[i], 0.0, 1.0, defaults[i], 0.01, key=f"w6_{i}")
           for i in range(8)]
    w_exp = np.array(raw) / sum(raw)
    C_exp = topsis(X, w_exp, is_ben)
    w_ent = entropy_weights(X)
    C_ent = topsis(X, w_ent, is_ben)

    res = pd.DataFrame({
        "Vùng": REGION_VI, "C* chuyên gia": C_exp.round(4), "C* entropy": C_ent.round(4),
    })
    res["Hạng (CG)"] = res["C* chuyên gia"].rank(ascending=False).astype(int)
    res["Hạng (Ent)"] = res["C* entropy"].rank(ascending=False).astype(int)
    res = res.sort_values("Hạng (CG)").reset_index(drop=True)

    c1, c2 = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = res.sort_values("C* chuyên gia")
    y = np.arange(6)
    ax.barh(y - 0.2, order["C* chuyên gia"], 0.4, color=ACC1, label="Chuyên gia")
    ax.barh(y + 0.2, order["C* entropy"], 0.4, color=ACC3, label="Entropy")
    ax.set_yticks(y); ax.set_yticklabels([r[:18] for r in order["Vùng"]], fontsize=8)
    ax.set_xlabel("C*"); ax.legend(); ax.grid(axis="x", alpha=.3)
    ax.set_title("TOPSIS: Chuyên gia vs Entropy")
    c1.pyplot(fig)
    c2.dataframe(res, width='stretch', hide_index=True)
    c2.success(f"🥇 Dẫn đầu (chuyên gia): {res.iloc[0]['Vùng']}")

    section("Câu 6.4.2 — Trọng số khách quan (Entropy)")
    st.dataframe(pd.DataFrame({"Tiêu chí": clabels, "w Entropy": w_ent.round(4)}),
                 width='stretch', hide_index=True)

    section("Câu 6.4.3 — Độ nhạy w_AI")
    w_ai_range = np.arange(0.10, 0.45, 0.05)
    hm = []
    for wai in w_ai_range:
        wg = 0.10; rem = 1 - wai - wg
        base = np.array([0.10, 0.10, 0.15, 0.15, 0.15, 0.05])
        ws = base * (rem / base.sum())
        wf = np.insert(ws, 3, wai); wf = np.append(wf, wg)
        hm.append(topsis(X, wf, is_ben))
    hm = np.array(hm)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(w_ai_range))); ax.set_yticklabels([f"{w:.2f}" for w in w_ai_range])
    ax.set_xticks(range(6)); ax.set_xticklabels([r[:12] for r in REGION_VI], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("w_AI"); ax.set_title("C* theo trọng số AI Readiness")
    plt.colorbar(im, label="C*"); fig.tight_layout()
    st.pyplot(fig)

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- **Đông Nam Bộ & ĐB sông Hồng** thường dẫn đầu — ứng viên cho trung tâm AI quốc gia "
            "đầu tiên theo Quyết định 127/QĐ-TTg.\n"
            "- Trọng số Entropy (khách quan) đôi khi đảo thứ hạng các vùng giữa do phản ánh "
            "độ phân tán dữ liệu thay vì ưu tiên chính sách.\n"
            "- AI Readiness và Internet penetration tương quan cao → có thể gây trùng lặp tiêu chí; "
            "nên cân nhắc PCA hoặc gộp tiêu chí trước khi chạy TOPSIS.\n"
            "- Chọn **3 vùng** cho 3 trung tâm AI: 2 vùng dẫn đầu + 1 vùng đại diện miền Trung "
            "để cân bằng yếu tố địa-chính trị."
        )


# ============================================================
# BÀI 7 — NSGA-II PARETO
# ============================================================
@st.cache_data(show_spinner=False)
def _run_nsga(pop_size, n_gen, lam):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    beta_mat = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGION_CODE])
    D0 = np.array([38, 78, 55, 32, 82, 48], float)
    e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
    gamma_val = 0.002

    class Prob(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=24, n_obj=4, n_ieq_constr=20,
                             xl=np.zeros(24), xu=np.ones(24) * 12000)

        def _evaluate(self, x, out, *a, **k):
            X = x.reshape(6, 4)
            f1 = -(beta_mat * X).sum()
            sums = X.sum(axis=1)
            f2 = np.abs(sums - sums.mean()).mean()
            f3 = (e * (X[:, 0] + X[:, 1] + X[:, 2])).sum()
            f4 = (rho * X[:, 2]).sum() - (sig * X[:, 3]).sum()
            out["F"] = [f1, f2, f3, f4]
            g = [X.sum() - 50000]
            for r in range(6): g.append(5000 - X[r].sum())
            for r in range(6): g.append(X[r].sum() - 12000)
            g.append(12000 - X[:, 3].sum())
            Dn = D0 + gamma_val * X[:, 1]; Dmax = Dn.max()
            for r in range(6): g.append(lam * Dmax - Dn[r])
            out["G"] = np.array(g)

    res = minimize(Prob(), NSGA2(pop_size=pop_size),
                   get_termination("n_gen", n_gen), seed=42, verbose=False)
    return res.F, res.X


def page_bai7():
    st.markdown("## 🌐 Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)")
    st.caption("4 mục tiêu: tăng trưởng · bao trùm · môi trường · an ninh dữ liệu")

    c = st.columns(3)
    pop = c[0].select_slider("Pop size", [40, 60, 80, 100], 60)
    ngen = c[1].select_slider("Số thế hệ", [50, 100, 150, 200], 100)
    lam = c[2].slider("Hệ số công bằng λ", 0.5, 0.9, 0.6, 0.05)

    with st.spinner("Đang chạy NSGA-II..."):
        F, X = _run_nsga(pop, ngen, lam)

    section("Câu 7.4.1–7.4.2 — Tập nghiệm Pareto")
    c = st.columns(4)
    c[0].metric("Số nghiệm Pareto", len(F))
    c[1].metric("GDP gain max", f"{-F[:,0].min():,.0f}")
    c[2].metric("Gini min", f"{F[:,1].min():.1f}")
    c[3].metric("Phát thải min", f"{F[:,2].min():,.0f}")

    c1, c2 = st.columns(2)
    fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(-F[:, 0], F[:, 1], F[:, 2], c=F[:, 3], cmap="viridis", s=14, alpha=.8)
    ax.set_xlabel("GDP gain"); ax.set_ylabel("Gini/MAD"); ax.set_zlabel("Phát thải")
    ax.set_title("Pareto 3D (màu = rủi ro)")
    fig.colorbar(sc, shrink=.6, label="f4 rủi ro")
    c1.pyplot(fig)

    Fn = F.copy()
    for i in range(4):
        lo, hi = F[:, i].min(), F[:, i].max()
        Fn[:, i] = (F[:, i] - lo) / (hi - lo) if hi > lo else 0.5
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(len(F)):
        ax.plot(range(4), Fn[i], color=ACC3, alpha=.06)
    ax.plot(range(4), Fn.mean(0), color=ACC1, lw=2, label="Trung bình")
    ax.set_xticks(range(4)); ax.set_xticklabels(["GDP\n(min↓)", "Gini\n(min)", "Phát thải\n(min)", "Rủi ro\n(min)"])
    ax.set_ylabel("Chuẩn hóa [0,1]"); ax.legend(); ax.set_title("Parallel coordinates")
    c2.pyplot(fig)

    section("Câu 7.4.3 — Nghiệm thỏa hiệp (TOPSIS trên Pareto)")
    w = np.array([0.40, 0.25, 0.20, 0.15])
    lo, hi = F.min(0), F.max(0)
    rng = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    R = (F - lo) / rng
    V = R * w
    Ss = np.sqrt(((V - 0) ** 2).sum(1)); Sn = np.sqrt(((V - w) ** 2).sum(1))
    Cs = Sn / (Ss + Sn)
    best = int(np.argmax(Cs))
    bF = F[best]
    c = st.columns(4)
    c[0].metric("GDP gain", f"{-bF[0]:,.0f}")
    c[1].metric("Gini/MAD", f"{bF[1]:.1f}")
    c[2].metric("Phát thải", f"{bF[2]:,.0f}")
    c[3].metric("Rủi ro ròng", f"{bF[3]:,.0f}")
    bX = X[best].reshape(6, 4)
    st.dataframe(pd.DataFrame(bX.round(0), columns=ITEMS, index=REGION_VI),
                 width='stretch')

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Đường biên Pareto cho thấy **đánh đổi rõ rệt giữa tăng trưởng và bao trùm**: "
            "nghiệm GDP cao nhất thường có Gini xấu hơn.\n"
            "- Trọng số (0,40; 0,25; 0,20; 0,15) ưu tiên tăng trưởng — cần điều chỉnh tăng trọng số "
            "môi trường để phù hợp cam kết COP26.\n"
            "- NSGA-II cung cấp **tập lựa chọn** cho nhà hoạch định, không thay thế quyết định "
            "chính trị về việc chọn điểm nào trên đường biên."
        )


# ============================================================
# BÀI 8 — TỐI ƯU ĐỘNG
# ============================================================
@st.cache_data(show_spinner=False)
def _run_dyn(rho_disc):
    from scipy.optimize import minimize
    a, b, gd, dai, th = 0.33, 0.42, 0.10, 0.08, 0.07
    dK, dD, dAI, thH, mu = 0.05, 0.12, 0.15, 0.8, 0.02
    phi1, phi2, phi3, gcr = 0.003, 0.002, 0.004, 1.5
    T = 10
    K0, L0, D0, AI0, H0 = 27500., 53.9, 20.3, 86., 30.
    A0 = 12847.6 / (K0 ** a * L0 ** b * D0 ** gd * AI0 ** dai * H0 ** th)
    L = np.array([L0 * 1.009 ** t for t in range(T + 1)])

    def traj(u):
        IK, ID, IAI, IH = u[0::4], u[1::4], u[2::4], u[3::4]
        K = np.zeros(T + 1); D = np.zeros(T + 1); AI = np.zeros(T + 1)
        H = np.zeros(T + 1); A = np.zeros(T + 1); Y = np.zeros(T + 1); C = np.zeros(T)
        K[0], D[0], AI[0], H[0], A[0] = K0, D0, AI0, H0, A0
        for t in range(T):
            Y[t] = A[t] * K[t]**a * L[t]**b * D[t]**gd * AI[t]**dai * H[t]**th
            C[t] = Y[t] - IK[t] - ID[t] - IAI[t] - IH[t]
            if C[t] <= 0: return None
            K[t+1] = (1-dK)*K[t]+IK[t]; D[t+1] = (1-dD)*D[t]+ID[t]
            AI[t+1] = (1-dAI)*AI[t]+IAI[t]; H[t+1] = H[t]+thH*IH[t]-mu*H[t]
            A[t+1] = A[t]*(1+phi1*(D[t]/100)+phi2*(AI[t]/100)+phi3*(H[t]/100))
        Y[T] = A[T]*K[T]**a*L[T]**b*D[T]**gd*AI[T]**dai*H[T]**th
        return K, D, AI, H, Y, C, A

    def welfare(u):
        r = traj(u)
        if r is None or np.any(r[5] <= 0): return 1e15
        C = r[5]
        return -sum(rho_disc**t * (C[t]**(1-gcr)-1)/(1-gcr) for t in range(T))

    ti = 14000 * 0.15
    u0 = np.tile([ti*0.4, ti*0.25, ti*0.2, ti*0.15], T)
    cons = [{"type": "ineq", "fun": lambda u: (lambda r: -1e10 if r is None else min(r[5])-1)(traj(u))}]
    res = minimize(welfare, u0, method="SLSQP", bounds=[(0, None)]*(T*4),
                   constraints=cons, options={"maxiter": 600, "ftol": 1e-8})
    return traj(res.x), -res.fun, np.arange(2026, 2037)


def page_bai8():
    st.markdown("## ⏳ Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035")
    st.caption("Cobb-Douglas động · CRRA · SLSQP · quỹ đạo K, D, AI, H, Y, C")

    rho = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)
    with st.spinner("Đang tối ưu quỹ đạo..."):
        (K, D, AI, H, Y, C, A), W, years = _run_dyn(rho)

    section("Câu 8.3.1–8.3.2 — Quỹ đạo tối ưu")
    st.metric("Phúc lợi W*", f"{W:.3f}")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, data, title in [
        (axes[0, 0], K, "K (vốn vật chất)"), (axes[0, 1], D, "D (hạ tầng số %GDP)"),
        (axes[0, 2], AI, "AI (nghìn DN)"), (axes[1, 0], H, "H (nhân lực %)"),
        (axes[1, 2], A, "A (TFP)"),
    ]:
        ax.plot(years, data, "o-", color=ACC1, ms=4); ax.set_title(title); ax.grid(alpha=.3)
    axes[1, 1].plot(years, Y, "o-", color=ACC2, ms=4, label="Y (GDP)")
    axes[1, 1].plot(years[:10], C, "s-", color=ACC3, ms=4, label="C (tiêu dùng)")
    axes[1, 1].set_title("Y & C"); axes[1, 1].legend(); axes[1, 1].grid(alpha=.3)
    fig.suptitle("Quỹ đạo tối ưu 2026-2035", fontsize=13); fig.tight_layout()
    st.pyplot(fig)

    df = pd.DataFrame({"Năm": years, "K": K.round(0), "D": D.round(1), "AI": AI.round(1),
                       "H": H.round(1), "Y": Y.round(0)})
    df["C"] = list(C.round(0)) + [np.nan]
    st.dataframe(df, width='stretch', hide_index=True)

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Quỹ đạo đầu tư có xu hướng **front-loaded** với hạ tầng số & AI do hiệu ứng tích "
            "lũy TFP nội sinh: đầu tư sớm sinh lợi suất kép qua nhiều năm.\n"
            "- Đầu tư nhân lực H nên **đi trước hoặc đồng thời** với AI vì H là điều kiện hấp thụ "
            "công nghệ.\n"
            "- ρ thấp hơn (0,90) khiến chính phủ ưu tiên ngắn hạn → giảm đầu tư R&D/AI, lý giải "
            "vì sao nhiều chính phủ 'dưới đầu tư' vào công nghệ dài hạn."
        )


# ============================================================
# BÀI 9 — LAO ĐỘNG & AI
# ============================================================
def page_bai9():
    from scipy.optimize import linprog
    st.markdown("## 👷 Bài 9 — Tác động AI tới thị trường lao động")
    st.caption("LP tối đa hóa NetJob · ngưỡng đào tạo lại · luồng dịch chuyển lao động")

    sec = ["Nông-LT", "CN chế biến", "Xây dựng", "Bán buôn-bán lẻ",
           "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục-ĐT"]
    L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55], float)
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62], float)
    N = 8
    coeff = a1 - c1 * risk

    cap5 = st.toggle("Câu 9.4.4 — Thêm ràng buộc: không ngành nào mất > 5% lao động", value=False)

    cobj = np.concatenate([-coeff, -b1])
    A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
    A1b = np.concatenate([-np.ones(N), np.zeros(N)]).reshape(1, -1)
    A2 = np.zeros((N, 2 * N)); A3 = np.zeros((N, 2 * N))
    for i in range(N):
        A2[i, i] = -coeff[i]; A2[i, N + i] = -b1[i]
        A3[i, i] = c1[i] * risk[i]; A3[i, N + i] = -d1[i]
    A_ub = np.vstack([A1, A1b, A2, A3])
    b_ub = np.concatenate([[30000], [-9000], np.zeros(N), np.zeros(N)])
    if cap5:
        A4 = np.zeros((N, 2 * N))
        for i in range(N): A4[i, i] = c1[i] * risk[i]
        A_ub = np.vstack([A_ub, A4]); b_ub = np.concatenate([b_ub, 0.05 * L * 1e6])
    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (2 * N), method="highs")

    section("Câu 9.4.1 — Phân bổ tối ưu & NetJob")
    if res.success:
        xA, xH = res.x[:N], res.x[N:]
        NJ = coeff * xA + b1 * xH
        Disp = c1 * risk * xA
        df = pd.DataFrame({"Ngành": sec, "x_AI": xA.round(0), "x_H": xH.round(0),
                           "Displaced": Disp.round(0), "NetJob": NJ.round(0)})
        c1c, c2c = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sec, NJ, color=ACC2, edgecolor=EDGE)
        ax.set_ylabel("NetJob (việc làm)"); ax.grid(axis="y", alpha=.3)
        ax.set_title("NetJob ròng theo ngành")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        c1c.pyplot(fig)
        c2c.metric("Tổng NetJob", f"{-res.fun:,.0f}")
        c2c.dataframe(df, width='stretch', hide_index=True)
    else:
        st.error("Bài toán KHÔNG khả thi với ràng buộc hiện tại.")

    section("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu (CN chế biến)")
    i = 1
    ratio = c1[i] * risk[i] / d1[i]
    net = a1[i] - c1[i] * risk[i]
    xr = np.linspace(0, 30000, 100)
    xh_re = ratio * xr
    xh_nj = np.maximum(0, -net / b1[i] * xr)
    xh_min = np.maximum(xh_re, xh_nj)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xr, xh_re, "--", color=ACC1, label=f"Retrain: x_H ≥ {ratio:.3f}·x_AI")
    ax.plot(xr, xh_nj, "--", color=ACC3, label="NetJob ≥ 0")
    ax.fill_between(xr, xh_min, 30000, alpha=.2, color=ACC2, label="Vùng khả thi")
    ax.set_xlabel("x_AI (tỷ VND)"); ax.set_ylabel("x_H tối thiểu (tỷ VND)")
    ax.set_xlim(0, 30000); ax.set_ylim(0, 30000); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Ngưỡng đào tạo lại tối thiểu — CN chế biến")
    st.pyplot(fig)
    st.info(f"Mỗi 1 tỷ đầu tư AI cần **{ratio:.3f} tỷ** đào tạo lại để giữ năng lực retraining.")

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Ngành **CN chế biến & Bán buôn-bán lẻ** cần đầu tư đào tạo lại nhiều nhất do "
            "rủi ro tự động hóa cao và lực lượng lao động lớn.\n"
            "- Tài chính-Ngân hàng rủi ro thay thế cao (52%) nhưng tạo việc làm mới cũng cao → "
            "chiến lược: đầu tư song song AI và đào tạo chuyển đổi kỹ năng.\n"
            "- Ràng buộc **'tốc độ tự động hóa ≤ năng lực đào tạo lại'** chính là "
            "Displaced ≤ RetrainingCapacity trong mô hình."
        )


# ============================================================
# BÀI 10 — STOCHASTIC 2 GIAI ĐOẠN
# ============================================================
J10 = ["I", "D", "AI", "H"]
S10 = ["s1", "s2", "s3", "s4"]
P_S = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
BETA_BASE = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BETA_S = {
    ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
    ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
    ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
    ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10,
}
SCEN_NAMES = {"s1": "Lạc quan", "s2": "Cơ sở", "s3": "Bi quan", "s4": "Khủng hoảng"}


@st.cache_data(show_spinner=False)
def _run_stochastic():
    import pyomo.environ as pyo

    def get_solver():
        for nm in ("appsi_highs", "glpk", "cbc"):
            s = pyo.SolverFactory(nm)
            try:
                if s.available():
                    return s
            except Exception:
                continue
        return pyo.SolverFactory("glpk")

    def build(scenarios, betas, fixed_x=None):
        m = pyo.ConcreteModel()
        m.J = pyo.Set(initialize=J10); m.S = pyo.Set(initialize=scenarios)
        m.beta = pyo.Param(m.J, initialize=BETA_BASE)
        m.beta_s = pyo.Param(m.S, m.J, initialize={(s, j): betas[s, j] for s in scenarios for j in J10})
        m.p = pyo.Param(m.S, initialize={s: (P_S[s] if len(scenarios) > 1 else 1.0) for s in scenarios})
        if fixed_x is None:
            m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
            m.b1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000)
        else:
            m.x = pyo.Param(m.J, initialize=fixed_x)
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
        m.b2 = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) <= 15000)
        m.aic = pyo.Constraint(m.S, rule=lambda m, s: m.y[s, "AI"] <= 0.5 * (m.x["H"] if fixed_x is None else fixed_x["H"]))
        m.obj = pyo.Objective(
            expr=sum(m.beta[j] * m.x[j] for j in m.J)
            + sum(m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J) for s in m.S),
            sense=pyo.maximize)
        return m

    solver = get_solver()
    m_sp = build(S10, BETA_S); solver.solve(m_sp)
    x_sp = {j: pyo.value(m_sp.x[j]) for j in J10}
    Z_SP = pyo.value(m_sp.obj)
    y_sp = {s: {j: pyo.value(m_sp.y[s, j]) for j in J10} for s in S10}

    det = {}
    for s in S10:
        md = build([s], BETA_S); solver.solve(md)
        det[s] = pyo.value(md.obj)

    beta_avg = {j: sum(P_S[s] * BETA_S[s, j] for s in S10) for j in J10}
    m_ev = pyo.ConcreteModel()
    m_ev.J = pyo.Set(initialize=J10); m_ev.x = pyo.Var(m_ev.J, within=pyo.NonNegativeReals)
    m_ev.b = pyo.Constraint(expr=sum(m_ev.x[j] for j in m_ev.J) <= 65000)
    m_ev.obj = pyo.Objective(expr=sum(beta_avg[j] * m_ev.x[j] for j in m_ev.J), sense=pyo.maximize)
    solver.solve(m_ev)
    x_ev = {j: pyo.value(m_ev.x[j]) for j in J10}
    Z_EV = sum(BETA_BASE[j] * x_ev[j] for j in J10)
    for s in S10:
        mt = build([s], BETA_S, fixed_x=x_ev); solver.solve(mt)
        Z_EV += P_S[s] * sum(BETA_S[s, j] * pyo.value(mt.y[s, j]) for j in J10)

    Z_WS = sum(P_S[s] * det[s] for s in S10)
    return x_sp, y_sp, Z_SP, x_ev, Z_EV, Z_WS, det


def page_bai10():
    st.markdown("## 🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn")
    st.caption("Pyomo · here-and-now vs recourse · VSS & EVPI")

    st.markdown("**Cây kịch bản:**")
    st.dataframe(pd.DataFrame({
        "Kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
        "Tăng trưởng TG (%)": [3.5, 2.8, 1.5, 0.2],
        "FDI VN (tỷ USD)": [32.0, 27.0, 20.0, 12.0],
        "Xác suất": [0.30, 0.45, 0.20, 0.05],
    }), width='stretch', hide_index=True)

    try:
        with st.spinner("Đang giải mô hình Pyomo..."):
            x_sp, y_sp, Z_SP, x_ev, Z_EV, Z_WS, det = _run_stochastic()
    except Exception as e:
        st.error(f"Không giải được (cần solver GLPK/HiGHS/CBC): {e}")
        return

    section("Câu 10.5.1 — Quyết định first-stage tối ưu (SP)")
    c = st.columns(2)
    c[0].metric("Z*_SP", f"{Z_SP:,.0f}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(J10, [x_sp[j] for j in J10], color=ACC1, edgecolor=EDGE)
    ax.set_ylabel("nghìn tỷ VND"); ax.set_title("First-stage SP"); ax.grid(axis="y", alpha=.3)
    c[1].pyplot(fig)

    section("Câu 10.5.2–10.5.3 — VSS & EVPI")
    VSS = Z_SP - Z_EV; EVPI = Z_WS - Z_SP
    c = st.columns(3)
    c[0].metric("Z_EV (lời giải EV)", f"{Z_EV:,.0f}")
    c[1].metric("VSS = Z_SP − Z_EV", f"{VSS:,.0f}", f"{VSS/Z_SP*100:.2f}%")
    c[2].metric("EVPI = Z_WS − Z_SP", f"{EVPI:,.0f}", f"{EVPI/Z_SP*100:.2f}%")
    st.info("**VSS > 0** → tư duy xác suất khi quyết định có giá trị. "
            "**EVPI** → giá trị tối đa nếu có thông tin hoàn hảo về tương lai.")

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = list(SCEN_NAMES.values())
    ax.bar(sc, [det[s] for s in S10], color=[ACC2, ACC3, "#f59e0b", "#ef4444"])
    ax.set_ylabel("Z* xác định"); ax.set_title("Z* tối ưu theo từng kịch bản (wait-and-see)")
    ax.grid(axis="y", alpha=.3)
    st.pyplot(fig)

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Lời giải SP có xu hướng đầu tư **H (nhân lực) nhiều hơn** lời giải xác định, vì H "
            "đóng vai trò 'bảo hiểm' — hệ số hiệu quả cao ngay cả trong kịch bản khủng hoảng.\n"
            "- VSS dương cho thấy **bỏ qua bất định gây thiệt hại** → hoạch định chính sách VN nên "
            "tính đến rủi ro toàn cầu (độ mở thương mại ≈180% GDP).\n"
            "- COVID-19 và bão Yagi là minh chứng: đầu tư nhân lực số như một hàng hóa bảo hiểm "
            "giúp hấp thụ cú sốc tốt hơn."
        )


# ============================================================
# BÀI 11 — Q-LEARNING
# ============================================================
@st.cache_data(show_spinner=False)
def _train_qlearning(n_episodes):
    import gymnasium as gym
    from gymnasium import spaces

    class Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
            self.T = 10
            self.alloc = {
                0: np.array([.70, .10, .10, .10]), 1: np.array([.40, .25, .15, .20]),
                2: np.array([.25, .45, .15, .15]), 3: np.array([.20, .20, .45, .15]),
                4: np.array([.30, .20, .10, .40]),
            }
            self.w = np.array([.40, .25, .20, .15])

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.state = (np.array(options["state"]) if options and "state" in options
                          else self.np_random.integers(0, 3, 4))
            self.t = 0; self.K = 27500.; self.D = 20.3; self.AI = 86.; self.H = 30.
            self.Yp = 12847.6
            return self.state.copy(), {}

        def step(self, action):
            a = self.alloc[action]; bud = 2100.
            self.K = .95 * self.K + a[0] * bud
            self.D = .88 * self.D + a[1] * bud * .01
            self.AI = .85 * self.AI + a[2] * bud * .05
            self.H = self.H + .8 * a[3] * bud * .01 - .02 * self.H
            A = 33.70 * (1 + .003*(self.D/100) + .002*(self.AI/100) + .004*(self.H/100))**self.t
            L = 53.9 * 1.009**self.t
            Y = A * self.K**.33 * L**.42 * self.D**.10 * self.AI**.08 * self.H**.07
            dg = (Y - self.Yp) / self.Yp
            du = max(0, -dg * .5)
            cyber = (self.AI / (self.H + 1)) * .01
            emis = (self.K + self.AI) * .0001
            r = self.w[0]*dg*100 - self.w[1]*du*100 - self.w[2]*cyber - self.w[3]*emis
            self.Yp = Y; self.t += 1
            gl = 0 if dg < .03 else (1 if dg < .06 else 2)
            dl = 0 if self.D < 25 else (1 if self.D < 35 else 2)
            al = 0 if self.AI < 100 else (1 if self.AI < 200 else 2)
            hl = 0 if self.H < 35 else (1 if self.H < 50 else 2)
            self.state = np.array([gl, dl, al, hl])
            return self.state.copy(), r, self.t >= self.T, False, {}

    env = Env()
    Q = np.zeros((3, 3, 3, 3, 5))
    hist = []
    for ep in range(n_episodes):
        s, _ = env.reset(); tot = 0; eps = max(.05, 1 - ep / (n_episodes / 2))
        while True:
            a = env.action_space.sample() if np.random.rand() < eps else int(np.argmax(Q[tuple(s)]))
            s2, r, done, _, _ = env.step(a)
            Q[tuple(s) + (a,)] += .1 * (r + .95 * np.max(Q[tuple(s2)]) * (1 - done) - Q[tuple(s) + (a,)])
            tot += r; s = s2
            if done: break
        hist.append(tot)

    def ev(fn, n=300):
        rs = []
        for _ in range(n):
            s, _ = env.reset(); t = 0
            while True:
                s, r, d, _, _ = env.step(fn(s)); t += r
                if d: break
            rs.append(t)
        return np.mean(rs), np.std(rs)

    pol = {
        "π* (Q-learning)": ev(lambda s: int(np.argmax(Q[tuple(s)]))),
        "Luôn Cân bằng": ev(lambda s: 1),
        "Luôn AI dẫn dắt": ev(lambda s: 3),
        "Random": ev(lambda s: np.random.randint(5)),
    }
    names = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm"]
    return Q, hist, pol, names


def page_bai11():
    st.markdown("## 🤖 Bài 11 — Q-learning cho chính sách kinh tế thích nghi")
    st.caption("MDP 81 trạng thái · 5 hành động · epsilon-greedy · so sánh rule-based")
    st.warning("⚠️ AI hỗ trợ ra quyết định, **không thay thế** trách nhiệm chính trị (Mục 11).")

    n_ep = st.select_slider("Số episode huấn luyện", [2000, 5000, 10000], 5000)
    with st.spinner("Đang huấn luyện Q-learning..."):
        Q, hist, pol, names = _train_qlearning(n_ep)

    section("Câu 11.3.3 — Chính sách π*(s) tại các trạng thái khởi đầu")
    test = [
        ([1, 1, 0, 1], "VN 2026 thực tế"),
        ([0, 0, 0, 2], "GDP thấp, D thấp, H cao"),
        ([2, 2, 2, 2], "Tất cả cao"),
        ([0, 1, 0, 0], "Sau khủng hoảng"),
        ([1, 0, 2, 1], "AI mạnh, D yếu"),
    ]
    rows = [{"Trạng thái": d, "π* (hành động)": names[int(np.argmax(Q[tuple(s)]))]} for s, d in test]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    section("Câu 11.3.4 — Learning curve & so sánh chính sách")
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    win = 200
    sm = np.convolve(hist, np.ones(win) / win, mode="valid")
    ax.plot(sm, color=ACC1); ax.set_xlabel("Episode"); ax.set_ylabel("Tổng phúc lợi")
    ax.set_title("Learning curve"); ax.grid(alpha=.3)
    c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = list(pol.keys()); ms = [pol[n][0] for n in ns]; ss = [pol[n][1] for n in ns]
    ax.bar(range(len(ns)), ms, yerr=ss, capsize=5,
           color=[ACC1, ACC3, ACC2, "#9ca3af"])
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Phúc lợi bình quân"); ax.set_title("So sánh chính sách"); ax.grid(axis="y", alpha=.3)
    c2.pyplot(fig)

    with st.expander("💬 Thảo luận chính sách"):
        st.markdown(
            "- Khi GDP thấp, D thấp, U cao, π* thường chọn hành động đẩy mạnh số hóa / cân bằng — "
            "tương ứng chiến lược **'quick win'**.\n"
            "- Khi mọi chỉ số đã cao, π* nghiêng về **'consolidation'** (cân bằng, ổn định).\n"
            "- π* nên được dùng như **công cụ tham mưu** trong quy trình hoạch định, kết quả phải "
            "được hội đồng chính sách thẩm định trước khi áp dụng."
        )


# ============================================================
# BÀI 12 — AIDEOM-VN TÍCH HỢP (6 MODULE → 4 TAB)
# ============================================================
SCENARIOS_12 = {
    "S1 Truyền thống": {"K": .70, "D": .10, "AI": .10, "H": .10},
    "S2 Số hóa nhanh": {"K": .25, "D": .45, "AI": .15, "H": .15},
    "S3 AI dẫn dắt": {"K": .20, "D": .20, "AI": .45, "H": .15},
    "S4 Bao trùm số": {"K": .30, "D": .20, "AI": .10, "H": .40},
    "S5 Tối ưu cân bằng": {"K": .25, "D": .25, "AI": .30, "H": .20},
}


@st.cache_data(show_spinner=False)
def _m1_forecast():
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    K0, L0, D0, AI0, H0, A0 = 27500, 53.9, 20.3, 86, 30, 33.70
    T, ba = 4, 3000

    def fc(al):
        K, D, AI, H, A = K0, D0, AI0, H0, A0
        tr = [A * K**a * L0**b * D**g * AI**d * H**th]
        for t in range(T):
            K = .95*K + al["K"]*ba; D = .88*D + al["D"]*ba*.01
            AI = .85*AI + al["AI"]*ba*.05; H = H + .8*al["H"]*ba*.01 - .02*H
            A = A*(1 + .003*(D/100) + .002*(AI/100) + .004*(H/100))
            L = L0 * 1.009**(t + 1)
            tr.append(A * K**a * L**b * D**g * AI**d * H**th)
        return tr
    return {n: fc(al) for n, al in SCENARIOS_12.items()}, list(range(2026, 2031))


def page_bai12():
    st.markdown("## 🇻🇳 Bài 12 — Nguyên mẫu AIDEOM-VN (tích hợp 6 module)")
    st.caption("Đồ án tổng kết · 6 module liên kết · dashboard 5 kịch bản chính sách")

    tabs = st.tabs([
        "📊 Tổng quan (M1·M2)",
        "🗺️ Phân bổ (M3·M4)",
        "⚖️ Rủi ro & Đa mục tiêu (M5)",
        "🎛️ So sánh kịch bản (M6)",
    ])

    # ---------- TAB 1: M1 dự báo + M2 TOPSIS ----------
    with tabs[0]:
        st.markdown("#### M1 — Dự báo kinh tế Cobb-Douglas 2026-2030")
        gdp_fc, years = _m1_forecast()
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, tr in gdp_fc.items():
            ax.plot(years, tr, "o-", ms=4, label=name)
        ax.set_xlabel("Năm"); ax.set_ylabel("GDP (nghìn tỷ VND)")
        ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("GDP theo 5 kịch bản")
        st.pyplot(fig)
        df_fc = pd.DataFrame({n: np.round(tr, 0) for n, tr in gdp_fc.items()}, index=years).T
        df_fc.columns = [str(y) for y in years]
        st.dataframe(df_fc, width='stretch')

        st.markdown("#### M2 — Đánh giá sẵn sàng số (TOPSIS + Entropy)")
        reg = load_regions()
        criteria = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD",
                    "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct",
                    "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
        is_ben = np.array([True] * 7 + [False])
        X = reg[criteria].values.astype(float)
        Ce = topsis(X, np.array([.10, .10, .15, .20, .15, .15, .05, .10]), is_ben)
        Cn = topsis(X, entropy_weights(X), is_ben)
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 4))
        y = np.arange(6)
        ax.barh(y - .2, Ce, .4, color=ACC1, label="Chuyên gia")
        ax.barh(y + .2, Cn, .4, color=ACC3, label="Entropy")
        ax.set_yticks(y); ax.set_yticklabels([r[:16] for r in REGION_VI], fontsize=8)
        ax.legend(); ax.grid(axis="x", alpha=.3); ax.set_title("Sẵn sàng AI 6 vùng")
        c1.pyplot(fig)
        tb = pd.DataFrame({"Vùng": REGION_VI, "C* CG": Ce.round(4), "C* Ent": Cn.round(4)})
        tb["Hạng"] = tb["C* CG"].rank(ascending=False).astype(int)
        c2.dataframe(tb.sort_values("Hạng"), width='stretch', hide_index=True)

    # ---------- TAB 2: M3 LP + M4 lao động ----------
    with tabs[1]:
        st.markdown("#### M3 — Tối ưu phân bổ ngân sách ngành-vùng (LP)")
        mat, Z = _solve_lp4(50000, True, 0.6)
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(6)); ax.set_yticklabels([r[:16] for r in REGION_VI], fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(ITEMS)
        ax.set_title(f"Phân bổ LP (Z*={Z:,.0f})")
        for i in range(6):
            for j in range(4):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > 7000 else "black")
        plt.colorbar(im, shrink=.8); fig.tight_layout()
        c1.pyplot(fig)
        c2.metric("Z* (GDP gain)", f"{Z:,.0f} tỷ")
        c2.metric("Tổng ngân sách", f"{mat.sum():,.0f} tỷ")
        c2.dataframe(pd.DataFrame(mat.round(0), columns=ITEMS, index=REGION_CODE),
                     width='stretch')

        st.markdown("#### M4 — Mô phỏng thị trường lao động (NetJob)")
        from scipy.optimize import linprog
        sec = ["Nông-LT", "CN chế biến", "Xây dựng", "Bán buôn", "Tài chính",
               "Logistics", "CNTT", "Giáo dục"]
        risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
        a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
        b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55], float)
        c1v = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
        d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62], float)
        N = 8; coeff = a1 - c1v * risk
        cobj = np.concatenate([-coeff, -b1])
        A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
        A2 = np.zeros((N, 2 * N)); A3 = np.zeros((N, 2 * N))
        for i in range(N):
            A2[i, i] = -coeff[i]; A2[i, N + i] = -b1[i]
            A3[i, i] = c1v[i] * risk[i]; A3[i, N + i] = -d1[i]
        r = linprog(cobj, A_ub=np.vstack([A1, A2, A3]),
                    b_ub=np.concatenate([[30000], np.zeros(N), np.zeros(N)]),
                    bounds=[(0, None)] * (2 * N), method="highs")
        NJ = coeff * r.x[:N] + b1 * r.x[N:]
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(sec, NJ, color=ACC2, edgecolor=EDGE)
        ax.set_ylabel("NetJob"); ax.grid(axis="y", alpha=.3); ax.set_title("NetJob ròng theo ngành")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        st.pyplot(fig)
        st.metric("Tổng NetJob", f"{-r.fun:,.0f} việc làm")

    # ---------- TAB 3: M5 NSGA + Stochastic ----------
    with tabs[2]:
        st.markdown("#### M5 — NSGA-II đa mục tiêu + Stochastic")
        with st.spinner("Đang chạy NSGA-II..."):
            F, X = _run_nsga(60, 100, 0.6)
        w = np.array([.40, .25, .20, .15])
        lo, hi = F.min(0), F.max(0); rng = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        V = ((F - lo) / rng) * w
        Cs = np.sqrt(((V - w) ** 2).sum(1)) / (np.sqrt((V ** 2).sum(1)) + np.sqrt(((V - w) ** 2).sum(1)))
        best = int(np.argmax(Cs))
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sc = ax.scatter(-F[:, 0], F[:, 1], c=F[:, 2], cmap="viridis", s=14, alpha=.8)
        ax.scatter(-F[best, 0], F[best, 1], c=ACC1, s=160, marker="*", zorder=5, label="Thỏa hiệp")
        ax.set_xlabel("GDP gain"); ax.set_ylabel("Gini/MAD"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Pareto (màu = phát thải)"); fig.colorbar(sc, shrink=.7)
        c1.pyplot(fig)
        c2.markdown("**Nghiệm thỏa hiệp (TOPSIS):**")
        c2.metric("GDP gain", f"{-F[best,0]:,.0f}")
        c2.metric("Gini/MAD", f"{F[best,1]:.1f}")
        c2.metric("Phát thải", f"{F[best,2]:,.0f}")

        st.markdown("**Stochastic (Pyomo) — quyết định first-stage dưới bất định**")
        try:
            x_sp, y_sp, Z_SP, x_ev, Z_EV, Z_WS, det = _run_stochastic()
            cc = st.columns(4)
            cc[0].metric("Z*_SP", f"{Z_SP:,.0f}")
            cc[1].metric("VSS", f"{Z_SP - Z_EV:,.0f}")
            cc[2].metric("EVPI", f"{Z_WS - Z_SP:,.0f}")
            cc[3].metric("x_H first-stage", f"{x_sp['H']:,.0f}")
        except Exception as e:
            st.info(f"Bỏ qua phần stochastic (thiếu solver): {e}")

    # ---------- TAB 4: M6 dashboard so sánh ----------
    with tabs[3]:
        st.markdown("#### M6 — Dashboard so sánh 5 kịch bản chính sách 2030")
        gdp_fc, years = _m1_forecast()
        summary = pd.DataFrame({
            "Kịch bản": list(gdp_fc.keys()),
            "GDP 2030 (nghìn tỷ)": [round(tr[-1], 0) for tr in gdp_fc.values()],
            "Tăng trưởng TB (%/năm)": [round(((tr[-1] / tr[0]) ** (1 / 4) - 1) * 100, 2)
                                       for tr in gdp_fc.values()],
        })
        summary = summary.sort_values("GDP 2030 (nghìn tỷ)", ascending=False).reset_index(drop=True)
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cols = [ACC1, ACC3, ACC2, "#f59e0b", "#a855f7"]
        ax.barh(summary["Kịch bản"], summary["GDP 2030 (nghìn tỷ)"], color=cols, edgecolor=EDGE)
        ax.set_xlabel("GDP 2030 (nghìn tỷ VND)"); ax.grid(axis="x", alpha=.3)
        ax.set_title("Xếp hạng kịch bản theo GDP 2030"); ax.invert_yaxis()
        c1.pyplot(fig)
        c2.dataframe(summary, width='stretch', hide_index=True)
        best_name = summary.iloc[0]["Kịch bản"]
        c2.success(f"🏆 Kịch bản tốt nhất 2030: **{best_name}**")

        st.markdown("**🚨 Cảnh báo & khuyến nghị:**")
        st.markdown(
            "- Kịch bản **AI dẫn dắt / Tối ưu cân bằng** thường cho GDP 2030 cao nhất nhờ "
            "hiệu ứng TFP nội sinh của đầu tư AI và số hóa.\n"
            "- **S4 Bao trùm số** đánh đổi tăng trưởng lấy công bằng vùng và an sinh — phù hợp "
            "khi ưu tiên giảm bất bình đẳng.\n"
            "- Khuyến nghị: lấy **S5 Tối ưu cân bằng** làm xương sống, bổ sung ràng buộc công bằng "
            "vùng (M3) và đào tạo lại lao động (M4) để giảm thiểu rủi ro xã hội."
        )

    with st.expander("📐 Kiến trúc 6 module AIDEOM-VN"):
        st.dataframe(pd.DataFrame({
            "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "Tên": ["Dự báo kinh tế", "Đánh giá sẵn sàng số", "Tối ưu phân bổ",
                    "Mô phỏng lao động", "Đánh giá rủi ro", "Dashboard ra QĐ"],
            "Kỹ thuật chính": ["Cobb-Douglas (Bài 1)", "TOPSIS + entropy (Bài 6)",
                               "LP (Bài 4) + Dynamic (Bài 8)", "LP NetJob (Bài 9)",
                               "NSGA-II (Bài 7) + SP (Bài 10)", "Streamlit / 4 tab"],
            "Tab": ["Tab 1", "Tab 1", "Tab 2", "Tab 2", "Tab 3", "Tab 4"],
        }), width='stretch', hide_index=True)


# ============================================================
# ĐIỀU HƯỚNG (giao diện mới — chọn cấp độ rồi chọn bài)
# ============================================================
NAV = {
    "🏠 Trang chủ": [("Trang chủ", "home")],
    "🟢 Dễ (Bài 1–3)": [
        ("Bài 1 · Cobb-Douglas + AI", "b1"),
        ("Bài 2 · LP ngân sách số", "b2"),
        ("Bài 3 · Priority 10 ngành", "b3"),
    ],
    "🟡 Trung bình (Bài 4–6)": [
        ("Bài 4 · LP ngành-vùng", "b4"),
        ("Bài 5 · MIP 15 dự án", "b5"),
        ("Bài 6 · TOPSIS 6 vùng", "b6"),
    ],
    "🟠 Khá khó (Bài 7–9)": [
        ("Bài 7 · NSGA-II Pareto", "b7"),
        ("Bài 8 · Tối ưu động", "b8"),
        ("Bài 9 · Lao động & AI", "b9"),
    ],
    "🔴 Khó (Bài 10–12)": [
        ("Bài 10 · Stochastic SP", "b10"),
        ("Bài 11 · Q-learning RL", "b11"),
        ("Bài 12 · AIDEOM tích hợp", "b12"),
    ],
}

ROUTES = {
    "home": page_home, "b1": page_bai1, "b2": page_bai2, "b3": page_bai3,
    "b4": page_bai4, "b5": page_bai5, "b6": page_bai6, "b7": page_bai7,
    "b8": page_bai8, "b9": page_bai9, "b10": page_bai10, "b11": page_bai11,
    "b12": page_bai12,
}

# Thanh điều hướng ngang trên cùng
c1, c2 = st.columns([1, 2])
with c1:
    group = st.selectbox("Cấp độ", list(NAV.keys()), index=0)
with c2:
    options = NAV[group]
    if len(options) == 1:
        choice_key = options[0][1]
        st.selectbox("Bài", [options[0][0]], index=0, disabled=True)
    else:
        labels = [o[0] for o in options]
        picked = st.selectbox("Bài", labels, index=0)
        choice_key = dict(zip(labels, [o[1] for o in options]))[picked]

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 14px;'>",
            unsafe_allow_html=True)

ROUTES.get(choice_key, page_home)()

st.markdown(
    "<hr style='border:none;border-top:1px solid #e2e8f0;margin:28px 0 8px;'>"
    "<div style='text-align:center;color:#94a3b8;font-size:.82rem;'>"
    "AIDEOM-VN · Trần Dương Nhi · Bài tập lớn: Các mô hình ra quyết định</div>",
    unsafe_allow_html=True,
)
