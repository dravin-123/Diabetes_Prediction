import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, bmi, age]])
    input_data = scaler.transform(input_data)  # Scale input
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.error("The model predicts that the person is **Diabetic**.")
    else:
        st.success("The model predicts that the person is **Non-Diabetic**.")
