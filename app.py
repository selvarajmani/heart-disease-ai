import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("heart_disease_dataset.csv")

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=80, max_depth=20, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Heart Risk Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Check your heart health risk by entering the details below.")

st.markdown("### 👤 Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restecg = st.selectbox("ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.2, step=0.1)
    slope = st.selectbox("ST Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

st.markdown("---")

if st.button("🔍 Predict Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]

    st.markdown("### 🩺 Prediction Result")
    if prediction == 1:
        st.error("⚠️ **High Risk Detected** – Please consult a cardiologist.")
    else:
        st.success("✅ **Low Risk** – No major indicators of heart disease.")
