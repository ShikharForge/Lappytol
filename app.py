"""
app.py — Laptop Price Predictor (Streamlit)
-------------------------------------------
Passes all inputs through the same src/predict.py pipeline used during training.
No preprocessing logic is duplicated here.
"""

import streamlit as st
import joblib
from pathlib import Path
from src.predict import predict_price

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load model metadata ────────────────────────────────────────────────────
@st.cache_resource
def load_metadata():
    meta_path = Path("models/feature_metadata.joblib")
    if not meta_path.exists():
        st.error("Model artifacts not found. Run `python src/train_final.py` first.")
        st.stop()
    return joblib.load(meta_path)

metadata = load_metadata()
test_r2   = metadata["test_metrics"]["r2"]
test_mae  = metadata["test_metrics"]["mae"]
test_rmse = metadata["test_metrics"]["rmse"]
cv_r2     = metadata["cv_metrics"]["r2"]
cv_mae    = metadata["cv_metrics"]["mae"]
cv_rmse   = metadata["cv_metrics"]["rmse"]

# ── Dropdown options from training data ───────────────────────────────────
COMPANIES = [
    "Acer", "Apple", "Asus", "Chuwi", "Dell", "Fujitsu", "Google",
    "HP", "Huawei", "LG", "Lenovo", "MSI", "Mediacom", "Microsoft",
    "Razer", "Samsung", "Toshiba", "Vero", "Xiaomi",
]
TYPENAMES = [
    "Notebook", "Ultrabook", "Gaming", "2 in 1 Convertible",
    "Workstation", "Netbook",
]
RAM_OPTIONS = ["2GB", "4GB", "6GB", "8GB", "12GB", "16GB", "24GB", "32GB", "64GB"]
INCHES_OPTIONS = [
    10.1, 11.3, 11.6, 12.0, 12.3, 12.5, 13.0, 13.3, 13.5, 13.9,
    14.0, 14.1, 15.0, 15.4, 15.6, 17.0, 17.3, 18.4,
]
OPSYS_OPTIONS = [
    "Windows 10", "Linux", "No OS", "macOS", "Chrome OS",
    "Windows 7", "Other OS",
]
SCREEN_OPTIONS = [
    "Full HD 1920x1080",
    "IPS Panel Full HD 1920x1080",
    "1366x768",
    "IPS Panel Full HD / Touchscreen 1920x1080",
    "Full HD / Touchscreen 1920x1080",
    "1600x900",
    "Touchscreen 1366x768",
    "Quad HD+ / Touchscreen 3200x1800",
    "IPS Panel 4K Ultra HD 3840x2160",
    "4K Ultra HD 3840x2160",
    "IPS Panel 4K Ultra HD / Touchscreen 3840x2160",
    "IPS Panel Retina Display 2560x1600",
    "IPS Panel 1366x768",
    "Touchscreen 2560x1440",
    "1440x900",
    "IPS Panel Retina Display 2304x1440",
    "IPS Panel Quad HD+ / Touchscreen 3200x1800",
    "4K Ultra HD / Touchscreen 3840x2160",
    "Touchscreen 2256x1504",
]
CPU_OPTIONS = [
    "Intel Core i5 7200U 2.5GHz",
    "Intel Core i5 8250U 1.6GHz",
    "Intel Core i5 6200U 2.3GHz",
    "Intel Core i5 7300HQ 2.5GHz",
    "Intel Core i5 7300U 2.6GHz",
    "Intel Core i5 6300U 2.4GHz",
    "Intel Core i5 6300HQ 2.3GHz",
    "Intel Core i5 1.6GHz",
    "Intel Core i7 7700HQ 2.8GHz",
    "Intel Core i7 7500U 2.7GHz",
    "Intel Core i7 8550U 1.8GHz",
    "Intel Core i7 6500U 2.5GHz",
    "Intel Core i7 6700HQ 2.6GHz",
    "Intel Core i7 6600U 2.6GHz",
    "Intel Core i7 6820HK 2.7GHz",
    "Intel Core i7 6820HQ 2.7GHz",
    "Intel Core i7 7600U 2.8GHz",
    "Intel Core i7 7820HK 2.9GHz",
    "Intel Core i7 7820HQ 2.9GHz",
    "Intel Core i7 7Y75 1.3GHz",
    "Intel Core i3 6006U 2GHz",
    "Intel Core i3 7100U 2.4GHz",
    "Intel Core i3 6006U 2.0GHz",
    "Intel Core i3 7130U 2.7GHz",
    "Intel Core i3 6100U 2.3GHz",
    "Intel Core M 6Y75 1.2GHz",
    "Intel Celeron Dual Core N3350 1.1GHz",
    "Intel Celeron Dual Core N3060 1.6GHz",
    "Intel Celeron Dual Core N3050 1.6GHz",
    "Intel Celeron Dual Core 3205U 1.5GHz",
    "Intel Celeron Quad Core N3450 1.1GHz",
    "Intel Pentium Quad Core N4200 1.1GHz",
    "Intel Pentium Quad Core N3710 1.6GHz",
    "Intel Atom x5-Z8350 1.44GHz",
    "Intel Atom x5-Z8550 1.44GHz",
    "AMD A9-Series 9420 3GHz",
    "AMD A6-Series 9220 2.5GHz",
    "AMD A8-Series 7410 2.2GHz",
    "AMD A12-Series 9720P 3.6GHz",
]
GPU_OPTIONS = [
    "Intel HD Graphics 620",
    "Intel HD Graphics 520",
    "Intel UHD Graphics 620",
    "Intel HD Graphics 500",
    "Intel HD Graphics 400",
    "Intel HD Graphics",
    "Intel HD Graphics 515",
    "Intel HD Graphics 615",
    "Intel HD Graphics 505",
    "Intel HD Graphics 405",
    "Intel Iris Plus Graphics 640",
    "Intel HD Graphics 6000",
    "Nvidia GeForce GTX 1050",
    "Nvidia GeForce GTX 1050 Ti",
    "Nvidia GeForce GTX 1060",
    "Nvidia GeForce GTX 1070",
    "Nvidia GeForce GTX 1080",
    "Nvidia GeForce GTX 960M",
    "Nvidia GeForce GTX 980M",
    "Nvidia GeForce GTX 950M",
    "Nvidia GeForce 940MX",
    "Nvidia GeForce GT 940MX",
    "Nvidia GeForce 930MX",
    "Nvidia GeForce 930M",
    "Nvidia GeForce 920MX",
    "Nvidia GeForce 920M",
    "Nvidia GeForce MX150",
    "Nvidia GeForce MX130",
    "Nvidia Quadro M1200",
    "Nvidia Quadro M620",
    "AMD Radeon 530",
    "AMD Radeon 520",
    "AMD Radeon R5 M430",
    "AMD Radeon R5 M420",
    "AMD Radeon R5 M330",
    "AMD Radeon R7 M445",
    "AMD Radeon R5",
    "AMD Radeon R2",
    "AMD Radeon RX 580",
    "AMD Radeon R4 Graphics",
]
MEMORY_OPTIONS = [
    "256GB SSD",
    "512GB SSD",
    "128GB SSD",
    "1TB SSD",
    "1TB HDD",
    "500GB HDD",
    "2TB HDD",
    "128GB SSD +  1TB HDD",
    "256GB SSD +  1TB HDD",
    "512GB SSD +  1TB HDD",
    "256GB SSD +  2TB HDD",
    "512GB SSD +  2TB HDD",
    "256GB SSD +  500GB HDD",
    "1TB SSD +  1TB HDD",
    "32GB Flash Storage",
    "64GB Flash Storage",
    "128GB Flash Storage",
    "256GB Flash Storage",
    "512GB Flash Storage",
    "16GB Flash Storage",
    "64GB Flash Storage +  1TB HDD",
    "32GB SSD",
    "64GB SSD",
    "16GB SSD",
    "1.0TB Hybrid",
]

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #c8cce0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
/* Header */
[data-testid="stHeader"] { background-color: #0f1117; }

/* Cards */
.card {
    background: #1a1d27;
    border: 1px solid #2c3057;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Section titles */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #7c83fd;
    margin-bottom: 0.8rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* Price display */
.price-box {
    background: linear-gradient(135deg, #1a1d27 0%, #1e2240 100%);
    border: 2px solid #7c83fd;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.price-label {
    font-size: 0.9rem;
    color: #9199c2;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.price-value {
    font-size: 2.8rem;
    font-weight: 800;
    color: #7c83fd;
    letter-spacing: -0.02em;
}

/* Metric chips */
.metric-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.metric-chip {
    background: #12141f;
    border: 1px solid #2c3057;
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    font-size: 0.82rem;
    color: #9199c2;
}
.metric-chip strong { color: #e8eaf6; }

/* Spec summary */
.spec-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1e2240;
    font-size: 0.87rem;
}
.spec-key { color: #9199c2; }
.spec-val { color: #e8eaf6; font-weight: 500; }

/* Divider */
hr { border-color: #2c3057; margin: 1.2rem 0; }

/* Streamlit select/input overrides */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label { color: #9199c2 !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("# 💻 Laptop Price Predictor")
st.markdown(
    "<p style='color:#9199c2;margin-top:-0.5rem;'>Estimate laptop prices in INR using "
    "a tuned XGBoost model trained on 1,303 laptops.</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Laptop Specifications</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        company   = st.selectbox("Brand",   COMPANIES, index=COMPANIES.index("Dell"))
        typename  = st.selectbox("Category", TYPENAMES, index=TYPENAMES.index("Notebook"))
        inches    = st.selectbox("Screen Size", INCHES_OPTIONS, index=INCHES_OPTIONS.index(15.6))
        screen    = st.selectbox("Screen Resolution", SCREEN_OPTIONS, index=0)
        opsys     = st.selectbox("Operating System", OPSYS_OPTIONS, index=0)

    with col2:
        cpu     = st.selectbox("CPU", CPU_OPTIONS, index=0)
        ram     = st.selectbox("RAM", RAM_OPTIONS, index=RAM_OPTIONS.index("8GB"))
        memory  = st.selectbox("Storage", MEMORY_OPTIONS, index=0)
        gpu     = st.selectbox("GPU", GPU_OPTIONS, index=0)
        weight_val = st.slider(
            "Weight (kg)", min_value=0.5, max_value=5.0,
            value=1.9, step=0.01, format="%.2f kg",
        )

    submitted = st.form_submit_button(
        "🔍  Predict Price",
        use_container_width=True,
        type="primary",
    )

# ── Prediction ─────────────────────────────────────────────────────────────
if submitted:
    weight_str = f"{weight_val:.2f}kg"

    input_data = {
        "Company":          company,
        "TypeName":         typename,
        "Inches":           float(inches),
        "ScreenResolution": screen,
        "Cpu":              cpu,
        "Ram":              ram,
        "Memory":           memory,
        "Gpu":              gpu,
        "OpSys":            opsys,
        "Weight":           weight_str,
    }

    try:
        result = predict_price(input_data, return_details=True)
        price  = result["predicted_price_inr"]

        # ── Price display ──────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        formatted_price = f"₹{price:,.0f}"
        st.markdown(
            f"""
            <div class="price-box">
                <div class="price-label">Estimated Price</div>
                <div class="price-value">{formatted_price}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Spec summary ───────────────────────────────────────────────────
        st.markdown(
            '<div class="card">'
            '<div class="section-title">Entered Specifications</div>',
            unsafe_allow_html=True,
        )
        spec_items = [
            ("Brand",      company),
            ("Category",   typename),
            ("Screen",     f"{inches}\"  ·  {screen}"),
            ("CPU",        cpu),
            ("RAM",        ram),
            ("Storage",    memory),
            ("GPU",        gpu),
            ("OS",         opsys),
            ("Weight",     weight_str),
        ]
        for key, val in spec_items:
            st.markdown(
                f'<div class="spec-row">'
                f'<span class="spec-key">{key}</span>'
                f'<span class="spec-val">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Log-price note ─────────────────────────────────────────────────
        st.caption(
            f"Model predicted log₁₊(Price) = {result['predicted_log_price']:.4f}, "
            f"then inverse-transformed using expm1 → ₹{price:,.0f}"
        )

    except ValueError as e:
        st.error(f"Input validation error: {e}")
    except FileNotFoundError as e:
        st.error(f"Model artifact not found: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Model information ──────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="card">
        <div style="font-size:0.9rem;color:#c8cce0;margin-bottom:0.8rem;">
            <strong style="color:#e8eaf6;">Final Model:</strong>
            XGBoost Regressor &nbsp;·&nbsp;
            <strong style="color:#e8eaf6;">Target:</strong> log₁(Price) → expm1 → INR
        </div>
        <div class="metric-row">
            <div class="metric-chip">CV R² <strong>{cv_r2}</strong></div>
            <div class="metric-chip">CV MAE <strong>₹{cv_mae:,.0f}</strong></div>
            <div class="metric-chip">CV RMSE <strong>₹{cv_rmse:,.0f}</strong></div>
            <div class="metric-chip">Test R² <strong>{test_r2}</strong></div>
            <div class="metric-chip">Test MAE <strong>₹{test_mae:,.0f}</strong></div>
            <div class="metric-chip">Test RMSE <strong>₹{test_rmse:,.0f}</strong></div>
        </div>
        <div style="margin-top:0.9rem;font-size:0.8rem;color:#9199c2;line-height:1.5;">
            Trained on 1,303 laptops · 5-fold CV · 80/20 train-test split ·
            Test set evaluated once · All metrics in INR
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "ℹ️  Prices are estimates. The model may be less accurate for rare brands, "
    "premium configurations (>₹2L), or hardware combinations not well-represented "
    "in the training data."
)
