import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
from streamlit_shap import st_shap

# 1. Page Configuration
st.set_page_config(
    page_title="NHS Health-Equity Optimizer",
    page_icon="🏥",
    layout="wide"
)

# 2. Custom CSS (NHS Branding)
st.markdown("""
    <style>
    .main { background-color: #f0f4f7; }
    .stMetric { 
        background-color: #ffffff; padding: 15px; border-radius: 10px; 
        border-left: 5px solid #005eb8; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Robust Data/Model Loader with Monkey Patching
@st.cache_resource
def load_assets():
    try:
        # Load the raw XGBoost model
        model = xgb.XGBClassifier()
        model.load_model('models/nhs_equity_model.json')
        
        # --- THE MONKEY PATCH & METADATA FIX ---
        # We extract the booster to bypass wrapper-level metadata issues
        booster = model.get_booster()
        
        # Fix 1: Manually define the estimator type for SHAP's internal checks
        booster._estimator_type = "classifier"
        
        # Fix 2: Clean the base_score string error (e.g., "[0.303]" -> 0.303)
        config = json.loads(booster.save_config())
        b_score_raw = config["learner"]["learner_model_param"]["base_score"]
        if isinstance(b_score_raw, str) and "[" in b_score_raw:
            clean_score = float(b_score_raw.strip("[]"))
            booster.set_param("base_score", clean_score)
        
        # Load Data
        df = pd.read_csv('data/nhs_patient_digital_twin_v1.csv')
        features = df.drop(columns=['DNA_Event'])
        
        # Fix 3: Set feature names explicitly on the booster for clear SHAP labels
        booster.feature_names = features.columns.tolist()
        
        # Initialize Explainer using the patched booster
        explainer = shap.TreeExplainer(booster)
        
        return model, df, explainer
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None

model, df, explainer = load_assets()

if model is not None:
    # 4. Sidebar: Navigation & Selection
    st.sidebar.image("https://www.nhs.uk/nhscms/img/nhs-logo.png", width=100)
    st.sidebar.header("Clinical Portal")
    
    features = df.drop(columns=['DNA_Event'])
    all_probs = model.predict_proba(features)[:, 1]

    mode = st.sidebar.radio("Analysis Mode", ["Single Patient Lookup", "Find Highest Risk Patient"])
    
    if mode == "Find Highest Risk Patient":
        patient_id = int(np.argmax(all_probs))
        st.sidebar.warning(f"Highest Risk Identified: ID {patient_id}")
    else:
        patient_id = st.sidebar.number_input("Select Patient ID", 0, len(df)-1, 10)

    patient_row = features.iloc[[patient_id]]
    prob = all_probs[patient_id]

    # 5. Main UI Header
    st.title("🏥 NHS Health-Equity Optimizer")
    st.markdown("🔍 **Predictive Analytics for DNA (Did Not Attend) Risk Mitigation**")
    st.markdown("---")

    # 6. Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="DNA Probability", value=f"{prob:.1%}")
    with m2:
        risk_cat = "🔴 HIGH" if prob > 0.4 else "🟡 MED" if prob > 0.2 else "🟢 LOW"
        st.metric("Risk Category", risk_cat)
    with m3:
        imd = int(patient_row['IMD_Decile'].values[0])
        st.metric("IMD Decile", imd, delta="High Deprivation" if imd <= 3 else "Moderate", delta_color="inverse")

    # 7. SHAP and Profile Columns
    st.markdown("---")
    col_profile, col_shap = st.columns([1, 1.5])

    with col_profile:
        st.subheader("Patient Profile Summary")
        # Transpose for better readability on narrow columns
        st.dataframe(patient_row.T.rename(columns={patient_id: 'Value'}))

    with col_shap:
        st.subheader("Risk Factor Explanation (SHAP)")
        try:
            # Generate SHAP values
            shap_values_obj = explainer(patient_row)
            
            # Final Safety: Force base_values to float64 to prevent plotting errors
            if not isinstance(shap_values_obj.base_values[0], (float, np.float64)):
                shap_values_obj.base_values = np.array([0.5], dtype=np.float64)
            
            # Render waterfall
            st_shap(shap.plots.waterfall(shap_values_obj[0]), height=400)
        except Exception as e:
            st.error(f"SHAP Plotting Error: {e}")

    # 8. Prescriptive Intervention Engine
    st.markdown("---")
    st.subheader("📋 Prescriptive Intervention")

    try:
        # Extract top driver from SHAP
        vals = shap_values_obj.values[0]
        top_feature_idx = np.argmax(np.abs(vals))
        top_feature_name = features.columns[top_feature_idx]

        if top_feature_name == 'Distance_KM':
            st.info("**Strategy:** Transport Barrier Detected. **Action:** Auto-send 'NHS Volunteer Transport' booking link.")
        elif top_feature_name == 'IMD_Decile':
            st.warning("**Strategy:** Socio-Economic Barrier Detected. **Action:** Schedule Social Prescriber call to discuss childcare/work conflicts.")
        elif 'Mental Health' in top_feature_name:
            st.error("**Strategy:** Clinical Sensitivity. **Action:** Assign 5-minute 'Human-in-the-loop' reminder call 24 hours prior.")
        else:
            st.success("**Strategy:** Standard Procedure. **Action:** Automated SMS reminder only.")
    except:
        st.write("Intervention logic unavailable due to SHAP initialization issues.")

    st.caption("NHS RAP Principles | Python 3.13 | Booster-Mixin Patch v1.6")
