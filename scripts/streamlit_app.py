import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import google.generativeai as genai

# Load the trained LightGBM model
MODEL_PATH = 'models/LightGBM_shelf_life_model.pkl'
model = joblib.load(MODEL_PATH)

# Define product options (should match training data)
PRODUCTS = ['Yogurt', 'Milk', 'Cheese', 'Bread', 'Juice']

st.title('Predictive Shelf Life Analytics')
st.write('Forecast the remaining shelf life of products based on environmental and product data.')

# User input
product = st.selectbox('Product', PRODUCTS)
manufacturing_date = st.date_input('Manufacturing Date', value=datetime.date(2023, 6, 1), min_value=datetime.date(2023, 1, 1), max_value=datetime.date(2023, 7, 1))
initial_shelf_life = st.number_input('Initial Shelf Life (days)', min_value=7, max_value=60, value=30)
storage_temp = st.number_input('Storage Temperature (°C)', min_value=-5.0, max_value=25.0, value=8.0)
storage_humidity = st.number_input('Storage Humidity (%)', min_value=20.0, max_value=100.0, value=60.0)
days_in_transit = st.number_input('Days in Transit', min_value=1, max_value=15, value=3)

# Feature engineering for input
product_age = (datetime.date(2023, 7, 1) - manufacturing_date).days
product_onehot = [1 if product == p else 0 for p in PRODUCTS]
temp_humidity_interaction = storage_temp * storage_humidity

# Prepare input DataFrame
input_data = pd.DataFrame([
    [initial_shelf_life, storage_temp, storage_humidity, days_in_transit, product_age] + product_onehot + [temp_humidity_interaction]
], columns=[
    'Initial_Shelf_Life', 'Storage_Temperature', 'Storage_Humidity', 'Days_in_Transit', 'Product_Age',
    'Product_Yogurt', 'Product_Milk', 'Product_Cheese', 'Product_Bread', 'Product_Juice',
    'Temp_Humidity_Interaction'
])

if st.button('Predict Remaining Shelf Life'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Remaining Shelf Life: {prediction:.1f} days')
    st.info('Reducing storage temperature, humidity, and transit time can help maximize shelf life and minimize spoilage.')

st.markdown('---')
st.markdown('**Business Impact:**')
st.markdown('This tool helps anticipate shelf life issues, optimize inventory, and reduce waste, leading to cost savings and improved sustainability.')

# --- Conversational Analytics (AI Assistant) ---

#st.markdown("---")
#st.header("Conversational Analytics (AI Assistant)")

# Set your Gemini API key
#genai.configure(api_key="AIzaSyCAJwDJDi69H_GjdxeTn-BjCEc8KbsW8jY")

# List available models
#st.subheader("Available Gemini Models")
#models = [m.name for m in genai.list_models()]
#st.write(models)

# Use the correct model name from the list
#model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# User input for the assistant
#user_question = st.text_input("Ask a question about shelf life analytics:")

#if st.button("Ask Gemini"):
#    if user_question.strip():
#       with st.spinner("Gemini is thinking..."):
#           try:
#               response = model.generate_content(user_question)
#               st.markdown("**Gemini's Answer:**")
#               st.write(response.text)
#           except Exception as e:
#               st.error("Gemini API quota exceeded. Please try again later or check your API usage limits.")
#    else:
#        st.warning("Please enter a question.")