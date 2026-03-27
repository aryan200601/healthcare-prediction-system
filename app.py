import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, columns
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details:")

# Inputs
age = st.number_input("Age", 20, 100)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
thalch = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Oldpeak (ST depression)")

sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
fbs = st.selectbox("Fasting Blood Sugar > 120", ["True", "False"])
exang = st.selectbox("Exercise Induced Angina", ["True", "False"])

# Predict
if st.button("Predict"):

    # Create input dictionary (IMPORTANT)
    input_dict = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "exang": exang
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    st.subheader("Result:")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({probability:.2f})")

pickle.dump(X.columns, open("columns.pkl", "wb"))
columns = pickle.load(open("columns.pkl", "rb"))