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

# 2. Custom CSS for NHS Branding
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
        # Load the model
        model = xgb.XGBClassifier()
        # Ensure your model is in a 'models/' folder on GitHub
        model.load_model('models/nhs_equity_model.json')
        
        # --- THE MONKEY PATCH & METADATA FIX ---
        booster = model.get_booster()
        
        # Fix 1: Inject the missing estimator type for SHAP
        booster._estimator_type = "classifier"
        
        # Fix 2: Clean the base_score string error (e.g., "[0.3038...]" -> 0.3038)
        config = json.loads(booster.save_config())
        b_score_raw = config["learner"]["learner_model_param"]["base_score"]
        if isinstance(b_score_raw, str) and "[" in b_score_raw:
            clean_score = float(b_score_raw.strip("[]"))
            booster.set_param("base_score", clean_score)
        
        # Load Data
        df = pd.read_csv('data/nhs_patient_digital_twin_v1.csv')
        features_df = df.drop(columns=['DNA_Event'])
        
        # Fix 3: Set feature names explicitly for clear SHAP plots
        booster.feature_names = features_df.columns.tolist()
        
        # Initialize Explainer using the patched booster
        explainer = shap.TreeExplainer(booster)
        
        return model, df, explainer
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None

model, df, explainer = load_assets()

if model is not None:
    # 4. Sidebar: Navigation
    st.sidebar.image("https://www.nhs.uk/nhscms/img/nhs-logo.png", width=100)
    st.sidebar.title("Clinical Portal")
    
    features = df.drop(columns=['DNA_Event'])
    # Calculate probabilities for all patients
    all_probs = model.predict_proba(features)[:, 1]

    mode = st.sidebar.radio("Analysis Mode", ["Single Patient Lookup", "Find Highest Risk Patient"])
    
    if mode == "Find Highest Risk Patient":
        patient_id = int(np.argmax(all_probs))
        st.sidebar.warning(f"Targeting Patient ID: {patient_id}")
    else:
        patient_id = st.sidebar.number_input("Select Patient ID", 0, len(df)-1, 10)

    # 5. Dashboard Header
    st.title("🏥 NHS Health-Equity Optimizer")
    st.markdown("🔍 **Predictive Analytics for DNA (Did Not Attend) Risk Mitigation**")
    st.divider()

    patient_row = features.iloc[[patient_id]]
    prob = all_probs[patient_id]

    # 6. Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("DNA Probability", f"{prob:.1%}")
    with m2:
        risk_cat = "🔴 HIGH" if prob > 0.4 else "🟡 MED" if prob > 0.2 else "🟢 LOW"
        st.metric("Risk Category", risk_cat)
    with m3:
        imd = int(patient_row['IMD_Decile'].values[0])
        st.metric("IMD Decile", imd, delta="High Deprivation" if imd <= 3 else "Moderate", delta_color="inverse")

    # 7. SHAP and Intervention logic
    col_left, col_right = st.columns([1, 1.5])

    with col_right:
        st.subheader("Decision Reasoning (XAI)")
        try:
            # Generate SHAP values
            shap_values_obj = explainer(patient_row)
            
            # Final Safety: Force base_values to float to prevent plotting string errors
            if not isinstance(shap_values_obj.base_values[0], (float, np.float64)):
                shap_values_obj.base_values = np.array([0.5], dtype=np.float64)
            
            st_shap(shap.plots.waterfall(shap_values_obj[0]), height=400)
        except Exception as e:
            st.error(f"SHAP Plotting Error: {e}")

    with col_left:
        st.subheader("📋 Clinical Intervention")
        try:
            # Use SHAP values to identify the top driver
            vals = shap_values_obj.values[0]
            top_driver_idx = np.argmax(np.abs(vals))
            top_driver_name = features.columns[top_driver_idx]

            if top_driver_name == 'Distance_KM':
                st.info("**Strategy:** Transport Support\\n\\n**Action:** Provision of NHS volunteer taxi voucher.")
            elif top_driver_name == 'IMD_Decile':
                st.warning("**Strategy:** Socio-Economic Support\\n\\n**Action:** Referral to Social Prescribing Link Worker.")
            else:
                st.success("**Strategy:** Standard Procedure\\n\\n**Action:** Automated SMS and email reminders.")
        except:
            st.write("Awaiting explanation data...")
            
        st.divider()
        st.write("**Patient Profile:**")
        st.dataframe(patient_row.T.rename(columns={patient_id: 'Value'}))

    st.caption("NHS RAP Principles | Python 3.13 | Booster-Mixin Patch v1.6")
else:
    st.warning("Please ensure your 'models/' and 'data/' folders are correctly placed in your repository.")
