import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')       
scaler = joblib.load('scaler.pkl')    

st.title("🔁 Customer Churn Prediction App")

st.markdown("Enter customer details below to predict whether they will churn.")

# Get user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ['Male', 'Female'])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)

# Encode gender: Male → 1, Female → 0
gender_encoded = 1 if gender == 'Male' else 0

# Combine into a DataFrame
input_df = pd.DataFrame([[age, gender_encoded, tenure, monthly_charges]],
                        columns=['Age', 'Gender', 'Tenure', 'MonthlyCharges'])

# Scale the input
input_scaled = scaler.transform(input_df)

st.write("Input DataFrame before scaling:")
st.write(input_df)

st.write("Scaled Input:")
st.write(input_scaled)

# Make prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.write("Model predicted probability for churn (class 1):", probability)


# Display result
if prediction == 1:
    st.error(f"🚨 This customer is likely to churn. (Probability: {probability:.2f})")
else:
    st.success(f"✅ This customer is not likely to churn. (Probability: {probability:.2f})")
