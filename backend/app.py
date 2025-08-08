# app.py

import streamlit as st
import joblib
import numpy as np
from chatbot_engine import get_health_advice

# Load trained model and encoders
model = joblib.load("models/disease_predictor_model.joblib")
le_gender = joblib.load("models/le_gender.joblib")
le_treatment = joblib.load("models/le_treatment.joblib")
le_disease = joblib.load("models/le_disease.joblib")

st.set_page_config(page_title="🧠 Intelligent Health Platform", layout="wide")

# App Header
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Intelligent Digital Health Platform</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Predict diseases and get AI-driven medical advice instantly</h5>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar: Patient Inputs
with st.sidebar:
    st.header("🩺 Patient Information")

    age = st.slider("Age", 1, 120, 30)
    gender = st.selectbox("Gender", le_gender.classes_)
    treatment = st.selectbox("Current Treatment", le_treatment.classes_)

    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
    blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=500, value=90)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)

    symptoms = st.text_area("📝 Symptoms", placeholder="e.g., chest pain, fatigue, frequent urination...")

    predict_button = st.button("🧬 Predict Disease")

# Main Section
if predict_button:
    try:
        # Encode categorical inputs
        gender_encoded = le_gender.transform([gender])[0]
        treatment_encoded = le_treatment.transform([treatment])[0]

        # Prepare input for model (only used features)
        input_data = np.array([[age, gender_encoded, treatment_encoded]])

        # Predict disease
        pred_encoded = model.predict(input_data)[0]
        pred_disease = le_disease.inverse_transform([pred_encoded])[0]

        # Display Prediction
        st.success("✅ Disease Prediction Successful!")

        st.markdown(f"""
        ### 🧾 Predicted Disease:  
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 10px;'>
        <h2 style='text-align: center; color: #FF4B4B;'>{pred_disease}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Summary Info
        st.markdown("### 🧍‍♂️ Patient Health Summary")
        st.markdown(f"""
        - **Age**: {age}  
        - **Gender**: {gender}  
        - **Current Treatment**: {treatment}  
        - **Blood Pressure**: {blood_pressure} mmHg  
        - **Blood Sugar**: {blood_sugar} mg/dL  
        - **Cholesterol**: {cholesterol} mg/dL  
        - **BMI**: {bmi}  
        """)

        if symptoms:
            st.markdown(f"🧪 **Symptoms**: _{symptoms}_")

        # Chatbot Advice
        with st.spinner("💬 Consulting AI Medical Assistant..."):
            advice = get_health_advice(pred_disease)

        st.markdown("### 💡 AI Health Recommendation")
        st.info(advice)

        # Future Expansion (Optional Section)
        with st.expander("👨‍⚕️ Simulate Doctor Visit Workflow"):
            st.markdown("""
            *Here you can simulate follow-ups, prescriptions, or schedule appointments in future versions.*
            - 📅 Schedule appointment
            - 📝 Prescription notes
            - 📊 View history
            """)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
