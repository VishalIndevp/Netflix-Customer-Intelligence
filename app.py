import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD FILES ----------------
model = joblib.load("model_XGB.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

num_cols = [
    'age', 'watch_hours', 'last_login_days',
    'monthly_fee', 'number_of_profiles', 'avg_watch_time_per_day'
]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ChurnLens · Netflix AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080a0f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8e8e8;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(229,9,20,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(229,9,20,0.08) 0%, transparent 60%),
        #080a0f !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(12, 14, 20, 0.95) !important;
    border-right: 1px solid rgba(229,9,20,0.15) !important;
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── SIDEBAR HEADER INJECT ── */
.sidebar-brand {
    padding: 28px 24px 20px;
    border-bottom: 1px solid rgba(229,9,20,0.15);
    margin-bottom: 8px;
}
.sidebar-brand h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 3px;
    color: #E50914;
    line-height: 1;
    margin-bottom: 4px;
}
.sidebar-brand p {
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(229,9,20,0.7);
    padding: 16px 24px 8px;
    display: block;
}

/* ── STREAMLIT LABEL OVERRIDES ── */
label, .stSlider label, .stSelectbox label,
[data-testid="stWidgetLabel"] p {
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    color: rgba(255,255,255,0.55) !important;
    text-transform: uppercase !important;
}

/* ── INPUT FIELDS ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s ease !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: rgba(229,9,20,0.5) !important;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #E50914, #ff6b6b) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: rgba(255,255,255,0.3) !important;
    font-size: 11px !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #E50914 0%, #b8070f 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 18px !important;
    letter-spacing: 3px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(229,9,20,0.3) !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(229,9,20,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── MAIN CONTENT ── */
.main-header {
    padding: 48px 0 32px;
    position: relative;
}
.eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #E50914;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(52px, 7vw, 88px);
    letter-spacing: 4px;
    line-height: 0.95;
    color: #fff;
    margin-bottom: 16px;
}
.hero-title span { color: #E50914; }
.hero-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.45);
    max-width: 480px;
    line-height: 1.6;
    font-weight: 300;
}

/* ── METRIC CARDS ── */
.metric-row { display: flex; gap: 16px; margin: 32px 0; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, transform 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #E50914, transparent);
}
.metric-card:hover {
    border-color: rgba(229,9,20,0.3);
    transform: translateY(-2px);
}
.metric-icon { font-size: 22px; margin-bottom: 10px; }
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    color: #fff;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── RESULT CARD ── */
.result-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 40px;
    position: relative;
    overflow: hidden;
    margin-top: 8px;
}
.result-wrapper.churn {
    border-color: rgba(229,9,20,0.35);
    background: rgba(229,9,20,0.04);
}
.result-wrapper.stay {
    border-color: rgba(0,210,110,0.35);
    background: rgba(0,210,110,0.04);
}
.result-wrapper::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    pointer-events: none;
}
.result-wrapper.churn::after {
    box-shadow: inset 0 0 60px rgba(229,9,20,0.08);
}
.result-wrapper.stay::after {
    box-shadow: inset 0 0 60px rgba(0,210,110,0.06);
}

.result-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.result-label.churn { color: #E50914; }
.result-label.stay { color: #00d26a; }

.result-headline {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 52px;
    letter-spacing: 2px;
    line-height: 1;
    margin-bottom: 24px;
}
.result-headline.churn { color: #ff4d57; }
.result-headline.stay { color: #00d26a; }

/* ── PROBABILITY BAR ── */
.prob-section { margin-top: 8px; }
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}
.prob-title {
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
}
.prob-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    letter-spacing: 1px;
}
.prob-value.churn { color: #ff4d57; }
.prob-value.stay { color: #00d26a; }

.prob-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    position: relative;
    overflow: hidden;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.prob-bar-fill.churn {
    background: linear-gradient(90deg, #b8070f, #E50914, #ff6b6b);
}
.prob-bar-fill.stay {
    background: linear-gradient(90deg, #00873f, #00d26a, #6effa5);
}
.prob-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 2s infinite;
}
@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* ── RISK SEGMENTS ── */
.risk-segments {
    display: flex;
    gap: 10px;
    margin-top: 28px;
    flex-wrap: wrap;
}
.risk-chip {
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.risk-chip.low  { background: rgba(0,210,106,0.12); color: #00d26a; border: 1px solid rgba(0,210,106,0.25); }
.risk-chip.med  { background: rgba(255,170,0,0.12); color: #ffaa00; border: 1px solid rgba(255,170,0,0.25); }
.risk-chip.high { background: rgba(229,9,20,0.12); color: #ff4d57; border: 1px solid rgba(229,9,20,0.25); }

/* ── INFO CARDS ── */
.info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 24px; }
.info-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
}
.info-card-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 6px;
}
.info-card-value {
    font-size: 16px;
    font-weight: 600;
    color: #fff;
}

/* ── DIVIDER ── */
.red-divider {
    height: 1px;
    background: linear-gradient(90deg, #E50914, transparent);
    margin: 32px 0;
    opacity: 0.4;
}

/* ── FOOTER ── */
.footer {
    margin-top: 64px;
    padding: 36px 0 28px;
    border-top: 1px solid rgba(255,255,255,0.07);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
}
.footer-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22px;
    letter-spacing: 4px;
    color: rgba(255,255,255,0.5);
}
.footer-name span { color: #E50914; }
.footer-links {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    justify-content: center;
}
.footer-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 99px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.03);
    text-decoration: none !important;
    color: rgba(255,255,255,0.5) !important;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.25s ease;
    cursor: pointer;
}
.footer-link:hover {
    border-color: rgba(229,9,20,0.5) !important;
    background: rgba(229,9,20,0.08) !important;
    color: #fff !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(229,9,20,0.2);
}
.footer-link svg { width: 15px; height: 15px; fill: currentColor; flex-shrink: 0; }
.footer-copy {
    font-size: 11px;
    color: rgba(255,255,255,0.2);
    letter-spacing: 1px;
}

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.stSidebar [data-testid="stVerticalBlock"] > div > div { padding: 0 16px !important; }
footer, [data-testid="stToolbar"] { display: none !important; }
[data-testid="collapsedControl"] { color: #E50914 !important; }
.stMarkdown p { line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h1>CHURNLENS</h1>
        <p>Netflix Customer Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-label">👤 Demographics</span>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown('<span class="section-label">📦 Subscription</span>', unsafe_allow_html=True)
    subscription_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium"])
    monthly_fee = st.number_input("Monthly Fee ($)", 0.0, 100.0, 10.0, step=0.5)
    number_of_profiles = st.slider("Number of Profiles", 1, 6, 2)
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "PayPal"])

    st.markdown('<span class="section-label">🌍 Location & Device</span>', unsafe_allow_html=True)
    region = st.selectbox("Region", ["Asia", "Europe", "America"])
    device = st.selectbox("Primary Device", ["Mobile", "TV", "Laptop", "Tablet"])

    st.markdown('<span class="section-label">🎬 Viewing Behavior</span>', unsafe_allow_html=True)
    watch_hours = st.number_input("Watch Hours / Month", 0.0, 500.0, 50.0, step=1.0)
    avg_watch_time_per_day = st.number_input("Avg Watch Time / Day (hrs)", 0.0, 24.0, 2.0, step=0.5)
    last_login_days = st.number_input("Days Since Last Login", 0, 365, 10)
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Drama", "Comedy", "Romance", "Sci-Fi"])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ ANALYZE CUSTOMER")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ── MAIN CONTENT ─────────────────────────────────────────────
col_main, col_right = st.columns([1.6, 1], gap="large")

with col_main:
    st.markdown("""
    <div class="main-header">
        <div class="eyebrow">AI-POWERED PREDICTION ENGINE</div>
        <div class="hero-title">PREDICT<br><span>CHURN</span><br>BEFORE IT<br>HAPPENS.</div>
        <p class="hero-sub">
            Advanced XGBoost model analyzing 12 behavioral and demographic signals
            to identify at-risk subscribers with clinical precision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics bar
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">XGB</div>
            <div class="metric-label">Model Type</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-value">12</div>
            <div class="metric-label">Features</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">&lt;10ms</div>
            <div class="metric-label">Inference</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🔒</div>
            <div class="metric-value">Live</div>
            <div class="metric-label">Status</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

    # ── RESULT AREA ──
    if predict_btn:
        # Build input
        input_data = pd.DataFrame({
            "age": [age], "gender": [gender], "subscription_type": [subscription_type],
            "watch_hours": [watch_hours], "last_login_days": [last_login_days],
            "region": [region], "device": [device], "monthly_fee": [monthly_fee],
            "payment_method": [payment_method], "number_of_profiles": [number_of_profiles],
            "avg_watch_time_per_day": [avg_watch_time_per_day], "favorite_genre": [favorite_genre]
        })
        input_data = pd.get_dummies(input_data, drop_first=True)
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            pct = round(probability * 100, 1)
            bar_width = round(probability * 100, 1)
            risk_level = "CRITICAL" if probability > 0.75 else "HIGH" if probability > 0.5 else "MODERATE"
            chip_cls = "high" if probability > 0.5 else "med"

            st.markdown(f"""
            <div class="result-wrapper churn">
                <div class="result-label churn">⚠ Churn Detected</div>
                <div class="result-headline churn">LIKELY TO<br>CANCEL</div>
                <div class="prob-section">
                    <div class="prob-header">
                        <span class="prob-title">Churn Probability</span>
                        <span class="prob-value churn">{pct}%</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill churn" style="width:{bar_width}%"></div>
                    </div>
                </div>
                <div class="risk-segments">
                    <span class="risk-chip {chip_cls}">Risk Level: {risk_level}</span>
                    <span class="risk-chip high">Action Required</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            pct = round((1 - probability) * 100, 1)
            bar_width = round((1 - probability) * 100, 1)
            st.markdown(f"""
            <div class="result-wrapper stay">
                <div class="result-label stay">✓ Retention Confirmed</div>
                <div class="result-headline stay">CUSTOMER<br>WILL STAY</div>
                <div class="prob-section">
                    <div class="prob-header">
                        <span class="prob-title">Loyalty Confidence</span>
                        <span class="prob-value stay">{pct}%</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill stay" style="width:{bar_width}%"></div>
                    </div>
                </div>
                <div class="risk-segments">
                    <span class="risk-chip low">Risk Level: LOW</span>
                    <span class="risk-chip low">Stable Subscriber</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 48px;
            text-align: center;
            color: rgba(255,255,255,0.2);
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">🔮</div>
            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 28px; letter-spacing: 3px; margin-bottom: 8px;">
                AWAITING INPUT
            </div>
            <div style="font-size: 13px; letter-spacing: 1px;">
                Fill in customer details and click <strong style="color:rgba(229,9,20,0.6)">ANALYZE CUSTOMER</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT COLUMN: Customer Summary ──
with col_right:
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        font-size: 10px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: rgba(229,9,20,0.7);
        font-weight: 600;
        margin-bottom: 14px;
    ">📋 Customer Snapshot</div>
    """, unsafe_allow_html=True)

    # Show customer summary cards
    details = [
        ("Age", f"{age} yrs"), ("Gender", gender),
        ("Plan", subscription_type), ("Region", region),
        ("Device", device), ("Genre", favorite_genre),
        ("Watch Hrs/Mo", f"{watch_hours:.0f}h"), ("Last Login", f"{last_login_days}d ago"),
        ("Profiles", str(number_of_profiles)), ("Monthly Fee", f"${monthly_fee:.2f}"),
    ]

    html = '<div class="info-grid">'
    for label, val in details:
        html += f"""
        <div class="info-card">
            <div class="info-card-label">{label}</div>
            <div class="info-card-value">{val}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 28px; padding: 20px; background: rgba(229,9,20,0.06);
        border: 1px solid rgba(229,9,20,0.15); border-radius: 14px;">
        <div style="font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
            color: rgba(229,9,20,0.7); font-weight: 600; margin-bottom: 10px;">ℹ How It Works</div>
        <p style="font-size: 13px; color: rgba(255,255,255,0.45); line-height: 1.7;">
            This model analyzes behavioral patterns — including login frequency, watch time, 
            subscription tier, and payment signals — to compute churn likelihood using XGBoost.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>

<style>
.footer-wrap {
    margin-top: 64px;
    padding: 40px 24px 32px;
    border-top: 1px solid rgba(255,255,255,0.07);
    text-align: center;
}
.footer-built {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 13px;
    letter-spacing: 4px;
    color: rgba(255,255,255,0.25);
    margin-bottom: 20px;
    text-transform: uppercase;
}
.footer-built span { color: #E50914; }
.social-row {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
}
.social-btn {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 11px 22px;
    border-radius: 99px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.6) !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-decoration: none !important;
    cursor: pointer;
    transition: all 0.25s ease;
}
.social-btn i { font-size: 15px; }
.social-btn:hover { transform: translateY(-3px); color: #fff !important; text-decoration: none !important; }
.social-btn.li:hover { background: rgba(0,119,181,0.15); border-color: rgba(0,119,181,0.5); box-shadow: 0 4px 20px rgba(0,119,181,0.2); }
.social-btn.li:hover i { color: #0077b5; }
.social-btn.x:hover  { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.4); box-shadow: 0 4px 20px rgba(255,255,255,0.1); }
.social-btn.gh:hover { background: rgba(110,84,148,0.15); border-color: rgba(110,84,148,0.5); box-shadow: 0 4px 20px rgba(110,84,148,0.2); }
.social-btn.gh:hover i { color: #c084fc; }
.social-btn.ig:hover { background: rgba(229,62,62,0.12); border-color: rgba(229,62,62,0.45); box-shadow: 0 4px 20px rgba(229,62,62,0.2); }
.social-btn.ig:hover i { color: #f77737; }
.footer-copy-line { font-size: 11px; color: rgba(255,255,255,0.18); letter-spacing: 1.5px; text-transform: uppercase; }
</style>

<div class="footer-wrap">
    <div class="footer-built">Join my journey and let's <span>explore together.</span></div>
    <div class="social-row">
        <a class="social-btn li" href="https://www.linkedin.com/in/vishal-singh-here/" target="_blank" rel="noopener noreferrer">
            <i class="fab fa-linkedin-in"></i> LinkedIn
        </a>
        <a class="social-btn x" href="https://x.com/vishalindev" target="_blank" rel="noopener noreferrer">
            <i class="fa-brands fa-x-twitter"></i> X (Twitter)
        </a>
        <a class="social-btn gh" href="https://github.com/VishalIndevp" target="_blank" rel="noopener noreferrer">
            <i class="fab fa-github"></i> GitHub
        </a>
        <a class="social-btn ig" href="https://www.instagram.com/vishalindev" target="_blank" rel="noopener noreferrer">
            <i class="fab fa-instagram"></i> Instagram
        </a>
    </div>
    <div class="footer-copy-line">© 2025 Vishal Singh</div>
</div>
""", unsafe_allow_html=True)