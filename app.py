import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("crime_model.pkl")

st.title("Crime Prediction System")

# User inputs
lat = st.number_input("Enter Latitude", value=37.77)
lon = st.number_input("Enter Longitude", value=-122.42)
hour = st.slider("Select Hour of the Day", 0, 23, 12)

# Predict crime type
if st.button("Predict Crime Type"):
    prediction = model.predict([[lat, lon, hour]])
    st.write(f"Predicted Crime Type: {prediction[0]}")