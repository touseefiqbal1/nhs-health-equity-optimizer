import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from streamlit_shap import st_shap

# =========================================================
# 1. Page Configuration
# =========================================================
st.set_page_config(
    page_title="NHS Health-Equity Optimizer",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# 2. Custom CSS (Dark NHS Theme - Readable)
# =========================================================
st.markdown("""
<style>

/* App Background */
.stApp {
    background: linear-gradient(180deg, #0b0f14 0%, #111827 100%);
    color: #ffffff;
}

/* Force all text readable */
html, body, [class*="css"] {
    color: #ffffff !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* Divider line */
hr {
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #1f2937 !important;
    padding: 18px !important;
    border-radius: 12px !important;
    border-left: 6px solid #005eb8 !important;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.35) !important;
}

/* Metric label */
div[data-testid="stMetric"] label {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Metric value */
div[data-testid="stMetric"] div {
    color: #ffffff !important;
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0b1220 !important;
    border-right: 2px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar radio / number input label text */
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: #005eb8 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    background-color: #111827 !important;
    border-radius: 12px !important;
    padding: 10px;
}

/* Alert boxes */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}

/* Make SHAP plot area readable */
iframe {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. Load Model + Data + SHAP Explainer
# =========================================================
@st.cache_resource
def load_assets():
    try:
        model = xgb.XGBClassifier()
        model.load_model("models/nhs_equity_model.json")

        df = pd.read_csv("data/nhs_patient_digital_twin_v1.csv")
        X = df.drop(columns=["DNA_Event"])

        explainer = shap.TreeExplainer(model)

        return model, df, X, explainer

    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None, None


model, df, features, explainer = load_assets()

# =========================================================
# 4. Main App
# =========================================================
if model is None:
    st.warning("Please ensure your 'models/' and 'data/' folders are correctly placed in your repository.")
    st.stop()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.image("https://www.nhs.uk/nhscms/img/nhs-logo.png", width=120)
st.sidebar.title("Clinical Portal")

# Model predictions for all patients
all_probs = model.predict_proba(features)[:, 1]

mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Patient Lookup", "Find Highest Risk Patient"]
)

if mode == "Find Highest Risk Patient":
    patient_id = int(np.argmax(all_probs))
    st.sidebar.warning(f"Targeting Patient ID: {patient_id}")
else:
    patient_id = st.sidebar.number_input(
        "Select Patient ID",
        min_value=0,
        max_value=len(df) - 1,
        value=10
    )


# =========================================================
# Dashboard Header
# =========================================================
st.title("🏥 NHS Health-Equity Optimizer")
st.markdown("🔍 **Predictive Analytics for DNA (Did Not Attend) Risk Mitigation**")
st.divider()


# =========================================================
# Patient Row + Prediction
# =========================================================
patient_row = features.iloc[[patient_id]]
prob = all_probs[patient_id]


# =========================================================
# Metrics Row
# =========================================================
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("DNA Probability", f"{prob:.1%}")

with m2:
    risk_cat = "🔴 HIGH" if prob > 0.4 else "🟡 MED" if prob > 0.2 else "🟢 LOW"
    st.metric("Risk Category", risk_cat)

with m3:
    imd = int(patient_row["IMD_Decile"].values[0])
    delta_text = "High Deprivation" if imd <= 3 else "Moderate / Low"
    st.metric("IMD Decile", imd, delta=delta_text, delta_color="inverse")


# =========================================================
# SHAP + Intervention Section
# =========================================================
col_left, col_right = st.columns([1, 1.6])

# ---------- SHAP Reasoning ----------
with col_right:
    st.subheader("Decision Reasoning (XAI)")

    try:
        shap_values_obj = explainer(patient_row)

        # Waterfall plot
        st_shap(shap.plots.waterfall(shap_values_obj[0]), height=420)

    except Exception as e:
        st.error(f"SHAP Plotting Error: {e}")


# ---------- Intervention Logic ----------
with col_left:
    st.subheader("📋 Clinical Intervention")

    try:
        vals = shap_values_obj.values[0]
        top_driver_idx = int(np.argmax(np.abs(vals)))
        top_driver_name = features.columns[top_driver_idx]

        if top_driver_name == "Distance_KM":
            st.info(
                "**Strategy:** Transport Support\n\n"
                "**Action:** Provide NHS volunteer taxi voucher or travel reimbursement."
            )

        elif top_driver_name == "IMD_Decile":
            st.warning(
                "**Strategy:** Socio-Economic Support\n\n"
                "**Action:** Refer to Social Prescribing Link Worker and local welfare services."
            )

        elif top_driver_name == "Age":
            st.info(
                "**Strategy:** Age-Specific Engagement\n\n"
                "**Action:** Offer assisted booking, phone reminders, and carer support."
            )

        else:
            st.success(
                "**Strategy:** Standard Reminder Procedure\n\n"
                "**Action:** Automated SMS + email reminder, plus optional phone follow-up."
            )

    except Exception:
        st.write("Awaiting explanation data...")

    st.divider()

    st.write("**Patient Profile:**")
    st.dataframe(patient_row.T.rename(columns={patient_id: "Value"}), use_container_width=True)


# =========================================================
# Footer
# =========================================================
st.caption("NHS RAP Principles | Streamlit Dashboard | XGBoost + SHAP XAI")
