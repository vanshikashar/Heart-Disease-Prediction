import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("Model/heart_model.pkl")
scaler = joblib.load("Model/scaler.pkl")
features = joblib.load("Model/features.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details below to predict the risk of heart disease.")

# -----------------------------
# User Inputs
# -----------------------------

age = st.slider("Age", 20, 100, 50)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    80,
    250,
    120
)

chol = st.number_input(
    "Cholesterol",
    100,
    600,
    240
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    60,
    250,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Old Peak",
    0.0,
    10.0,
    1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# -----------------------------
# Convert Inputs
# -----------------------------

sex = 1 if sex == "Male" else 0

data = {
    "age": age,
    "sex": sex,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "ca": ca
}

# Add all model features with default value 0
for feature in features:
    if feature not in data:
        data[feature] = 0

# One-hot encoded columns
cp_col = f"cp_{cp}"
if cp_col in data:
    data[cp_col] = 1

restecg_col = f"restecg_{restecg}"
if restecg_col in data:
    data[restecg_col] = 1

slope_col = f"slope_{slope}"
if slope_col in data:
    data[slope_col] = 1

thal_col = f"thal_{thal}"
if thal_col in data:
    data[thal_col] = 1

input_df = pd.DataFrame([data])

# Arrange columns in training order
input_df = input_df[features]

# Scale continuous columns
continuous = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

input_df[continuous] = scaler.transform(
    input_df[continuous]
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"**Probability of Heart Disease:** {probability:.2%}")

    st.progress(float(probability))

    st.info(
        "This prediction is based on a machine learning model and "
        "should not replace professional medical advice."
    )