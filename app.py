import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load the trained model and preprocessor
model = joblib.load("insurance_model.pkl")
preprocessor = joblib.load("insurance_preprocessor.pkl")

st.title("Insurance Charges Predictor")

st.write("Enter the patient information below to predict insurance charges:")

# --- Input Form ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# --- Predict on button click ---
if st.button("Predict Charges"):
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Preprocess + Predict
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)[0]

    st.success(f"Estimated Insurance Charges: ${prediction:,.2f}")

# --- Optional Static Example Plot ---
if st.checkbox("Show Example Prediction Plot"):
    # Simulated example data (just for UI purpose)
    actual = np.linspace(2000, 40000, 50)
    predicted = actual + np.random.normal(0, 3000, 50)

    df_plot = pd.DataFrame({
        "Actual Charges": actual,
        "Predicted Charges": predicted
    })

    fig = px.scatter(
        df_plot,
        x="Actual Charges",
        y="Predicted Charges",
        title="Example: Actual vs Predicted Charges",
        opacity=0.6,
        template="plotly_white"
    )

    fig.add_shape(
        type="line",
        x0=actual.min(),
        y0=actual.min(),
        x1=actual.max(),
        y1=actual.max(),
        line=dict(color="red", dash="dash")
    )

    st.plotly_chart(fig, use_container_width=True)
