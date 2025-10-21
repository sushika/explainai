import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import io
import streamlit as st
import pandas as pd
import shap
import pickle
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="ExplainAI — Day 2+", layout="wide")
st.title("ExplainAI — Global Feature Importance")

st.markdown("""
**How to use**
1) Upload a **CSV** (e.g., `breast_cancer.csv`).  
2) Upload a **trained model** `.pkl`.  
3) Choose an explainer to compute **global feature importance** with SHAP.
""")

# -----------------------------
# Sidebar explanation
# -----------------------------
with st.sidebar:
    st.header("ℹ️ What is SHAP?")
    st.markdown("""
**Plain English**
- SHAP explains *why* a model made a prediction by showing **how each feature pushed the result up or down**.
- Think of the model as a team decision; SHAP shows **how much each player (feature)** contributed.

**Every prediction can be written as:**  
`prediction ≈ base value + Σ feature contributions (SHAP values)`
""")

# -----------------------------
# Uploads
# -----------------------------
data_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
model_file = st.file_uploader("Upload trained model (.pkl)", type=["pkl"])
target_col = st.text_input("Target column (optional)", value="target")


def load_model_upload(buf):
    """Load model (pickle -> joblib fallback) with friendly errors."""
    try:
        return pickle.load(buf)
    except Exception:
        buf.seek(0)
        try:
            return joblib.load(buf)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()


# -----------------------------
# Main flow
# -----------------------------
if data_file and model_file:
    # Data
    df = pd.read_csv(data_file)
    st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Model
    model = load_model_upload(model_file)
    st.success(f"✅ Model loaded: `{type(model).__name__}`")

    # Features
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()

    # ---- Safe slider bounds for small datasets
    n_rows = len(X)
    if n_rows == 0:
        st.error("Your CSV appears to be empty.")
        st.stop()

    max_sample_cap = min(2000, n_rows)
    sample_min = min(50, max_sample_cap)
    sample_default = min(300, max_sample_cap)

    # Sampling slider
    max_rows = st.slider(
        "Sample rows for SHAP (lower = faster)",
        min_value=sample_min,
        max_value=max_sample_cap,
        value=sample_default,
        step=1
    )
    Xs = X.sample(max_rows, random_state=42) if n_rows > max_rows else X

    # Explainer
    explainer_choice = st.selectbox("Explainer", ["TreeExplainer", "KernelExplainer"])

    # Compute SHAP
    with st.spinner("Computing SHAP values..."):
        try:
            if explainer_choice == "TreeExplainer":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(Xs)  # new API -> Explanation object
            else:
                bg = Xs.sample(min(50, len(Xs)), random_state=0)
                explainer = shap.KernelExplainer(model.predict, bg)
                shap_values = explainer.shap_values(Xs, nsamples=100)  # legacy -> array/list
        except Exception as e:
            st.error(f"Failed to compute SHAP: {e}")
            st.stop()

    # -----------------------------
    # SHAP Plot (auto-scaling fonts, compact, PNG render)
    # -----------------------------
    st.subheader("Global Feature Importance")

    # Normalize SHAP output (works for new/old APIs)
    def _to_array(vals):
        if hasattr(vals, "values"):
            return vals.values
        if isinstance(vals, list):
            return vals[1] if len(vals) > 1 else vals[0]
        return vals

    vals = _to_array(shap_values)

    # How many features to display (safe bounds)
    n_features = Xs.shape[1]
    if n_features == 0:
        st.error("No feature columns found after preprocessing.")
        st.stop()

    feats_min = 1 if n_features == 1 else min(3, n_features)
    feats_max = min(25, n_features)
    feats_default = min(10, n_features)

    max_feats = st.slider(
        "Max features to display",
        min_value=feats_min,
        max_value=feats_max,
        value=feats_default,
        step=1
    )

    # Auto font-size based on number of displayed features
    def autosize_font(k, *, min_feats=5, max_feats=30, min_size=4, max_size=9):
        k = max(min_feats, min(max_feats, int(k)))
        ratio = (max_feats - k) / (max_feats - min_feats)
        return min_size + ratio * (max_size - min_size)

    f = autosize_font(max_feats)

    # Base figure size (compact & capped)
    width_in = 3.8
    height_in = min(max(2.0, 0.22 * max_feats + 0.5), 3.6)

    # Draw SHAP beeswarm
    plt.close("all")
    shap.summary_plot(
        vals, Xs,
        show=False,
        plot_type="dot",
        max_display=max_feats,
        plot_size=(width_in, height_in)
    )

    # Enforce font sizes (some SHAP versions override rcParams)
    fig = plt.gcf()
    ax = plt.gca()

    # Force ticks (both axes)
    for lbl in (ax.get_xticklabels() + ax.get_yticklabels()):
        lbl.set_fontsize(f)
    # Force any other text (axis labels, colorbar ticks, etc.)
    for txt in fig.findobj(match=plt.Text):
        txt.set_fontsize(f)

    fig.set_size_inches(width_in, height_in)
    plt.tight_layout(pad=0.35)

    # Render to PNG buffer and display at fixed pixel width (no Streamlit stretching)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=460, caption="SHAP Global Feature Importance")

    st.caption(
        "Each dot = one row. Color = feature value (red = high, blue = low). "
        "Further right = larger positive contribution to the model output."
    )
    st.info("Note: SHAP shows **feature influence**, not causal effects.")

else:
    st.caption("Upload a CSV and model to begin.")
