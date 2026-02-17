Project Overview
This project addresses the £1.2 billion annual cost of missed appointments (DNAs) within the UK National Health Service. Unlike traditional models that focus purely on clinical history, this Health-Equity Optimizer integrates Social Determinants of Health (SDoH).

By linking patient demographics with geographic deprivation data (ONS IMD), the model identifies patients at risk of missing appointments due to systemic barriers—such as transport poverty or financial instability—rather than just personal preference.

Core Features
Digital Twin Population: A high-fidelity synthetic dataset (n=5,000) generated to mirror West Yorkshire’s age and socio-economic distributions, ensuring GDPR compliance while maintaining research validity.

Explainable AI (XAI): Implements SHAP (SHapley Additive exPlanations) to move beyond "Black Box" predictions, providing clinicians with a visual "reasoning" for every risk score.

Prescriptive Engine: Automatically maps SHAP risk drivers to specific NHS interventions (e.g., Volunteer Transport, Social Prescribing, or Telehealth shifts).

RAP Compliant: Developed following Reproducible Analytical Pipelines (RAP) principles used by NHS England and the ONS.

Technical Stack
Model: XGBoost Classifier (Log-Loss Optimization)

Explainability: SHAP Waterfall & Summary Plots

Integration: ONS Indices of Multiple Deprivation (IMD) 2019/2024

Deployment: Streamlit for real-time Clinical Decision Support (CDS)

Why this Research Matters
In the 2026 NHS landscape, the shift toward Preventative Care requires models that are not just accurate, but fair. This tool demonstrates how Machine Learning can be used to promote Health Equity by identifying why a patient might struggle to attend, allowing the Trust to provide proactive support rather than punitive measures.
