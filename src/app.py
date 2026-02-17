import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from streamlit_shap import st_shap

# 1. Page Configuration
st.set_page_config(
    page_title="NHS Health-Equity Optimizer",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS to align with NHS Branding (Blues and Whites)
st.markdown("""
    <style>
    .main { background-color: #f0f4f7; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #005eb8; }
    </style>
    """, unsafe_allow_none=True)

# 2. Optimized Data/Model Loader
@st.cache_resource
def load_resources():
    try:
        # Load Model
        model = xgb.XGBClassifier()
        model.load_model('models/nhs_equity_model.json')
        
        # Load Data
        df = pd.read_csv('data/nhs_patient_digital_twin_v1.csv')
        return model, df
    except Exception as e:
        st.error(f"Error loading resources: {e}. Ensure 'models/' and 'data/' folders are present.")
        return None, None

model, df = load_resources()

if model is not None:
    # 3. Sidebar - Patient Discovery
    st.sidebar.image("https://www.nhs.uk/nhscms/img/nhs-logo.png", width=100)
    st.sidebar.title("Navigation")
    
    mode = st.sidebar.radio("Analysis Mode", ["Single Patient Lookup", "Population High-Risk Hunt"])
    
    X = df.drop(columns=['DNA_Event'])
    all_probs = model.predict_proba(X)[:, 1]

    if mode == "Population High-Risk Hunt":
        patient_id = np.argmax(all_probs)
        st.sidebar.info(f"Top Risk Patient identified at Index: {patient_id}")
    else:
        patient_id = st.sidebar.number_input("Enter Patient ID", 0, len(df)-1, 0)

    # 4. Main Dashboard UI
    st.title("🏥 NHS Health-Equity Optimizer")
    st.markdown("### Clinical Decision Support Tool for Appointment Attendance")
    
    patient_row = X.iloc[[patient_id]]
    prob = all_probs[patient_id]

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DNA Probability", f"{prob:.1%}")
    with col2:
        risk_level = "High" if prob > 0.4 else "Medium" if prob > 0.2 else "Low"
        st.metric("Risk Category", risk_level)
    with col3:
        imd = patient_row['IMD_Decile'].values[0]
        st.metric("IMD Decile", imd, delta="Most Deprived" if imd <= 3 else "Moderate")

    st.divider()

    # 5. SHAP Explainability Section
    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        st.subheader("Patient Demographics")
        st.dataframe(patient_row.T.rename(columns={patient_id: 'Value'}))
        
        # Prescriptive Intervention Logic
        st.subheader("📋 Prescribed Intervention")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(patient_row)
        
        # Find strongest driver
        top_driver_idx = np.argmax(shap_values.values[0])
        top_driver_name = X.columns[top_driver_idx]

        if top_driver_name == 'Distance_KM':
            st.warning("**Transport Intervention:** Auto-provision of NHS Volunteer Taxi voucher.")
        elif top_driver_name == 'IMD_Decile':
            st.error("**Socio-Economic Support:** Referral to Social Prescribing Link Worker for childcare/work flexibility.")
        elif 'type_Mental Health' in top_driver_name:
            st.info("**Enhanced Engagement:** Peer-support 'warm-call' reminder 48h prior.")
        else:
            st.success("**Standard Care:** Automated SMS and Email reminders.")

    with right_col:
        st.subheader("Why this prediction?")
        # Display SHAP Waterfall
        st_shap(shap.plots.waterfall(shap_values[0]), height=400)

    st.caption("Disclaimer: This is a research tool using synthetic data for health-equity modeling purposes.")
