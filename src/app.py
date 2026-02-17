import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
from streamlit_shap import st_shap

# 1. Page Configuration
st.set_page_config(page_title="NHS Health-Equity Optimizer", page_icon="🏥", layout="wide")

# 2. Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f4f7; }
    .stMetric { 
        background-color: #ffffff; padding: 15px; border-radius: 10px; 
        border-left: 5px solid #005eb8; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Robust Data/Model Loader with Metadata Patching
@st.cache_resource
def load_resources():
    try:
        # Load Model
        model = xgb.XGBClassifier()
        model.load_model('models/nhs_equity_model.json')
        
        # --- THE METADATA PATCH ---
        # We extract the booster and fix the base_score string error
        booster = model.get_booster()
        config = json.loads(booster.save_config())
        
        # Access the problematic base_score
        b_score_raw = config["learner"]["learner_model_param"]["base_score"]
        
        # If it's a string like "[0.303]", strip brackets and force to float
        if isinstance(b_score_raw, str) and "[" in b_score_raw:
            clean_score = float(b_score_raw.strip("[]"))
            booster.set_param("base_score", clean_score)
        
        # Load Data
        df = pd.read_csv('data/nhs_patient_digital_twin_v1.csv')
        X = df.drop(columns=['DNA_Event'])
        
        # Initialize Explainer using the patched booster
        # Passing the booster directly avoids the "not callable" error
        explainer = shap.TreeExplainer(booster)
        
        return model, df, explainer
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None

model, df, explainer = load_resources()

if model is not None:
    # 4. Sidebar & Navigation
    st.sidebar.image("https://www.nhs.uk/nhscms/img/nhs-logo.png", width=100)
    st.sidebar.title("Clinical Portal")
    
    X = df.drop(columns=['DNA_Event'])
    all_probs = model.predict_proba(X)[:, 1]

    mode = st.sidebar.radio("Analysis Mode", ["Single Patient Lookup", "Find Highest Risk Patient"])
    patient_id = int(np.argmax(all_probs)) if mode == "Find Highest Risk Patient" else st.sidebar.number_input("Enter Patient ID", 0, len(df)-1, 0)

    # 5. Dashboard Header
    st.title("🏥 NHS Health-Equity Optimizer")
    st.markdown("🔍 **Predictive Analytics for DNA (Did Not Attend) Risk Mitigation**")
    
    patient_row = X.iloc[[patient_id]]
    prob = all_probs[patient_id]

    # 6. Metrics
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("DNA Probability", f"{prob:.1%}")
    with m2: 
        risk_cat = "🔴 HIGH" if prob > 0.4 else "🟡 MED" if prob > 0.2 else "🟢 LOW"
        st.metric("Risk Category", risk_cat)
    with m3:
        imd = int(patient_row['IMD_Decile'].values[0])
        st.metric("IMD Decile", imd, delta="High Deprivation" if imd <= 3 else "Moderate", delta_color="inverse")

    st.divider()

    # 7. SHAP and Interventions
    col_left, col_right = st.columns([1, 1.5])

    with col_right:
        st.subheader("Decision Reasoning (XAI)")
        try:
            # Generate SHAP values
            shap_values_obj = explainer(patient_row)
            
            # Final Safety: Force base_values to float64 to prevent plotting errors
            if not isinstance(shap_values_obj.base_values[0], (float, np.float64)):
                shap_values_obj.base_values = np.array([0.5], dtype=np.float64)
            
            st_shap(shap.plots.waterfall(shap_values_obj[0]), height=400)
        except Exception as e:
            st.error(f"SHAP Plotting Error: {e}")

    with col_left:
        st.subheader("📋 Clinical Intervention")
        try:
            # Use the SHAP values to find the top driver
            vals = shap_values_obj.values[0]
            top_driver_idx = np.argmax(np.abs(vals))
            top_driver_name = X.columns[top_driver_idx]

            if top_driver_name == 'Distance_KM':
                st.info("**Strategy:** Transport Support\n**Action:** Provision of NHS volunteer driver.")
            elif top_driver_name == 'IMD_Decile':
                st.warning("**Strategy:** Socio-Economic Support\n**Action:** Referral to Social Prescriber.")
            else:
                st.success("**Strategy:** Standard Procedure\n**Action:** Automated SMS reminder.")
        except:
            st.write("Awaiting explanation data...")
            
        st.write("---")
        st.write("**Patient Demographics:**")
        st.dataframe(patient_row.T.rename(columns={patient_id: 'Value'}))

    st.caption("NHS RAP Principles | Python 3.13 | Booster-Metadata Patch v1.5")
