#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('rf_heart_model.pkl')

st.title("❤️ Heart Disease Risk Predictor")

st.write("Fill the details below to check if you're at risk for heart disease.")

# Example input fields (you can adjust to match your dataset)
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", [0, 1])  # 1 = Male, 0 = Female
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression", 0.0, 6.0)
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=normal; 2=fixed defect; 3=reversible)", [1, 2, 3])

# Prediction button
if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(user_input)[0]
    
    if prediction == 1:
        st.error("⚠️ You may be at risk for heart disease. Please consult a doctor.")
    else:
        st.success("✅ You seem to be at low risk for heart disease.")

