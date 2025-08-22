# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------- setup ----------
st.set_page_config(page_title="🏠 Housing Price Predictor", page_icon="🏡", layout="centered")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "house_model.pkl"

# ---------- load model ----------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    features = obj["features"]
    # optional, if you saved it in model.py (not required)
    train_rmse = obj.get("train_rmse", None)
    return model, features, train_rmse

try:
    model, FEATURES, TRAIN_RMSE = load_model()
except Exception as e:
    st.error(f"Could not load model at `{MODEL_PATH}`. Error: {e}")
    st.stop()

# ---------- styles ----------
st.markdown("""
<style>
/* page width tweaks */
.main .block-container{ padding-top:2.2rem; max-width: 900px; }
/* big title */
.big-title{ font-size: 2.2rem; font-weight:800; letter-spacing:.2px; }
/* soft cards */
.card{ background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
       border-radius: 16px; padding: 1rem 1.1rem; }
/* result badge */
.result{ background: linear-gradient(135deg, #0e7, #0bb);
         color: #fff; border-radius:14px; padding: 1rem 1.2rem; font-size: 1.05rem;
         border: 1px solid rgba(0,0,0,.1); }
/* labels */
.stSlider label, .stSelectbox label, .stNumberInput label, .stRadio label{ font-weight:600; }
/* helper text size */
small.hint{ opacity:.7; }
</style>
""", unsafe_allow_html=True)

# ---------- header ----------
st.markdown('<div class="big-title">🏡 Housing Price Prediction App</div>', unsafe_allow_html=True)
st.markdown("Enter a few details about the home and click **Predict Price**. "
            "We’ll use a trained regression model to estimate the price.")

# ---------- input form ----------
with st.form("predict_form", clear_on_submit=False):
    st.markdown("### 📋 Home details")

    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sq ft)", min_value=200, max_value=15000, value=2000, step=50, help="Total built-up area.")
        bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3, help="Count of bedrooms.")
        bathrooms = st.number_input("Number of Bathrooms", 1, 10, 2)
        stories = st.number_input("Number of Stories", 1, 5, 2)
        parking = st.slider("Parking Spaces", 0, 5, 1, help="Covered/open parking spots.")
        furnishingstatus = st.selectbox("Furnishing Status",
                                        ["furnished", "semi-furnished", "unfurnished"])

    with c2:
        mainroad = st.radio("Main Road Access?", ["yes", "no"], horizontal=True)
        guestroom = st.radio("Guest Room?", ["yes", "no"], horizontal=True)
        basement = st.radio("Basement?", ["yes", "no"], horizontal=True)
        hotwaterheating = st.radio("Hot Water Heating?", ["yes", "no"], horizontal=True)
        airconditioning = st.radio("Air Conditioning?", ["yes", "no"], horizontal=True)
        prefarea = st.radio("Preferred Area?", ["yes", "no"], horizontal=True)
        

    with st.expander("⚙️ Advanced (optional)"):
        currency = st.radio("Currency display", ["₹ INR", "$ USD"], horizontal=True, index=0)
        show_psf = st.checkbox("Show price per sq ft", False)

    submitted = st.form_submit_button("💰 Predict Price", use_container_width=True)

# ---------- prepare input ----------
def assemble_row():
    """
    Build a single-row DataFrame that matches the training feature columns (FEATURES).
    The LinearRegression model expects one-hot columns created with get_dummies(drop_first=True).
    """
    row = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        # one-hot booleans
        "mainroad_yes": 1 if mainroad == "yes" else 0,
        "guestroom_yes": 1 if guestroom == "yes" else 0,
        "basement_yes": 1 if basement == "yes" else 0,
        "hotwaterheating_yes": 1 if hotwaterheating == "yes" else 0,
        "airconditioning_yes": 1 if airconditioning == "yes" else 0,
        "prefarea_yes": 1 if prefarea == "yes" else 0,
        # furnishingstatus → base is "furnished"; we created dummies for the other two
        "furnishingstatus_semi-furnished": 1 if furnishingstatus == "semi-furnished" else 0,
        "furnishingstatus_unfurnished": 1 if furnishingstatus == "unfurnished" else 0,
    }
    X = pd.DataFrame([row])

    # ensure all training columns exist
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURES]  # same order as training
    return X

# ---------- prediction ----------
if submitted:
    X = assemble_row()
    y_hat = float(model.predict(X)[0])

    # currency formatting
    def money(x):
        if currency.startswith("$"):
            return f"$ {x:,.0f}"
        return f"₹ {x:,.0f}"

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown('<div class="result">🏠 Estimated House Price:<br>'
                    f'<span style="font-size:1.6rem; font-weight:800;">{money(y_hat)}</span>'
                    '</div>', unsafe_allow_html=True)

    with colB:
        psf = y_hat / max(area, 1)
        rmse_txt = f"± {money(TRAIN_RMSE)} (train RMSE)" if TRAIN_RMSE else "Model: Linear Regression"
        st.markdown(
            f"""<div class="card">
                <b>Details</b><br>
                • Area: {area:,} sq ft<br>
                • Bedrooms/Bathrooms: {bedrooms}/{bathrooms}<br>
                • Stories: {stories} &nbsp;• Parking: {parking}<br>
                • Main road: {mainroad}, AC: {airconditioning}, HW heating: {hotwaterheating}<br>
                • Furnishing: {furnishingstatus}, Pref. area: {prefarea}<br>
                {'• Price / sq ft: ' + money(psf) if show_psf else ''}
                <br><br><small class="hint">{rmse_txt}</small>
            </div>""",
            unsafe_allow_html=True,
        )

    # download prediction
    out = {
        "predicted_price": y_hat,
        "currency": currency,
        "area_sqft": area,
        "price_per_sqft": psf,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }
    csv = pd.DataFrame([out]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download result (CSV)", data=csv, file_name="prediction.csv", mime="text/csv")
    st.balloons()
