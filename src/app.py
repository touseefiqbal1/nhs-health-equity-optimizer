import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from streamlit_shap import st_shap

# 1. Page Configuration
st.set_page_config(page_title="NHS Health-Equity Optimizer", layout="wide")

# 2. Load Model and Data
@st.cache_resource
def load_assets():
    # Load the model you saved
    model = xgb.XGBClassifier()
    model.load_model('models/nhs_equity_model.json')
    
    # Load the data for reference/sampling
    df = pd.read_csv('data/nhs_patient_digital_twin_v1.csv')
    return model, df

model, df = load_assets()

# 3. Sidebar: Patient Selection
st.sidebar.header("Patient Selector")
patient_id = st.sidebar.number_input("Select Patient ID", 0, len(df)-1, 10)

# Extract features for the selected patient
features = df.drop(columns=['DNA_Event'])
target = df['DNA_Event']
patient_row = features.iloc[[patient_id]]

# 4. Main UI
st.title("🏥 NHS Health-Equity Optimizer")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Profile")
    st.write(patient_row)
    
    # Make Prediction
    prob = model.predict_proba(patient_row)[0][1]
    st.metric(label="DNA Probability", value=f"{prob:.1%}")
    
    if prob > 0.5:
        st.error("⚠️ High Risk of Missed Appointment")
    else:
        st.success("✅ Low Risk of Missed Appointment")

with col2:
    st.subheader("Risk Factor Explanation (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(patient_row)
    
    # Render the SHAP plot
    st_shap(shap.plots.waterfall(shap_values[0]), height=400)

# 5. Intervention Engine (The Prescriptive Layer)
st.markdown("---")
st.subheader("📋 Prescriptive Intervention")

# Logic to find the top driver from SHAP
top_feature_idx = np.argmax(shap_values.values[0])
top_feature_name = features.columns[top_feature_idx]

if top_feature_name == 'Distance_KM':
    st.info("**Strategy:** Transport Barrier Detected. **Action:** Auto-send 'NHS Volunteer Transport' booking link.")
elif top_feature_name == 'IMD_Decile':
    st.warning("**Strategy:** Socio-Economic Barrier Detected. **Action:** Schedule Social Prescriber call to discuss childcare/work conflicts.")
elif 'Mental Health' in top_feature_name:
    st.info("**Strategy:** Clinical Sensitivity. **Action:** Assign 5-minute 'Human-in-the-loop' reminder call 24 hours prior.")
else:
    st.success("**Strategy:** Standard Procedure. **Action:** Automated SMS reminder only.")
