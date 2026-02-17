import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# 1. Page Configuration
st.set_page_config(
    page_title="NHS Health-Equity Optimizer",
    page_icon="🏥",
    layout="wide"
)

# 2. Custom CSS (Corrected parameter: unsafe_allow_html)
st.markdown("""
    <style>
    .main { background-color: #f0f4f7; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #005eb8; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Robust Data/Model Loader
@st.cache_resource
def load_resources():
    try:
        model = xgb.XGBClassifier()
        # Ensure pathing matches your GitHub folder structure
        model.load_model('models/nhs_equity_model.json')
        
        # Explicitly set base_score to avoid SHAP ValueError
        model.get_booster().set_param('base_score', 0.5)
        
        df = pd.read_csv('data/nhs_patient_digital_twin_v1.csv')
        return model, df
    except Exception as e:
        st.error(f"Resource Load Error: {e}")
        return None, None

model, df = load_resources()

if model is not None:
    # 4. Sidebar Navigation
    st.sidebar.image("https://www.nhs.uk/nhscms/img/nhs-logo.png", width=100)
    st.sidebar.title("Clinical Portal")
    
    mode = st.sidebar.radio("Analysis Mode", ["Single Patient Lookup", "Find Highest Risk Patient"])
    
    X = df.drop(columns=['DNA_Event'])
    # Pre-calculate probabilities for the sidebar
    all_probs = model.predict_proba(X)[:, 1]

    if mode == "Find Highest Risk Patient":
        patient_id = int(np.argmax(all_probs))
        st.sidebar.warning(f"Highest Risk identified at ID: {patient_id}")
    else:
        patient_id = st.sidebar.number_input("Enter Patient ID", 0, len(df)-1, 0)

    # 5. Dashboard Header
    st.title("🏥 NHS Health-Equity Optimizer")
    st.markdown("🔍 **Predictive Analytics for DNA (Did Not Attend) Risk Mitigation**")
    
    patient_row = X.iloc[[patient_id]]
    prob = all_probs[patient_id]

    # 6. Top-Level Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("DNA Probability", f"{prob:.1%}")
    with m2:
        risk_cat = "🔴 HIGH" if prob > 0.4 else "🟡 MED" if prob > 0.2 else "🟢 LOW"
        st.metric("Risk Category", risk_cat)
    with m3:
        imd = int(patient_row['IMD_Decile'].values[0])
        st.metric("IMD Decile", imd, delta="High Deprivation" if imd <= 3 else "Moderate", delta_color="inverse")

    st.divider()

    # 7. SHAP and Interventions
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("📋 Clinical Intervention")
        
        # SHAP Logic with Safety Catch
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(patient_row)
        
        # Determine top driver for intervention text
        top_driver_idx = np.argmax(np.abs(shap_values.values[0]))
        top_driver_name = X.columns[top_driver_idx]

        if top_driver_name == 'Distance_KM':
            st.info("**Strategy:** Transport Support\n\n**Action:** Offer NHS volunteer driver or travel reimbursement.")
        elif top_driver_name == 'IMD_Decile':
            st.warning("**Strategy:** Socio-Economic Support\n\n**Action:** Referral to Social Prescriber for childcare/cost-of-living assistance.")
        elif 'Mental Health' in top_driver_name:
            st.error("**Strategy:** Clinical Sensitivity\n\n**Action:** 1-to-1 phone call reminder from a known clinical staff member.")
        else:
            st.success("**Strategy:** Standard Procedure\n\n**Action:** Digital SMS reminder 24 hours prior.")
            
        st.write("---")
        st.write("**Patient Data Summary:**")
        st.dataframe(patient_row.T.rename(columns={patient_id: 'Value'}))

    with col_right:
        st.subheader("Decision Reasoning (XAI)")
        # Display SHAP Waterfall
        st_shap(shap.plots.waterfall(shap_values[0]), height=400)
        st.caption("SHAP values explain how each feature pushed the probability away from the average.")

    st.markdown("---")
    st.caption("Internal Research Tool | NHS England Data Science Standards (Synthetic Data)")

else:
    st.warning("Please check your repository structure. Ensure `models/nhs_equity_model.json` exists.")
