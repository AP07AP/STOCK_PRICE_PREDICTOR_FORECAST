import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the trained LSTM model from the .h5 file
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('lstm_model.h5')
    return model

model = load_model()

# Set up the Streamlit app
st.title("Stock Price Prediction and Forecasting")
st.write("Enter the stock data below to predict the stock price.")

# User input fields for each feature
date = st.text_input("Date (YYYY-MM-DD)", value="2024-11-30")
close = st.number_input("Close Price", value=150.0)
high = st.number_input("High Price", value=152.0)
low = st.number_input("Low Price", value=148.0)
open_price = st.number_input("Open Price", value=149.0)
volume = st.number_input("Volume", value=5000000)
adjClose = st.number_input("Adjusted Close Price", value=150.0)
adjHigh = st.number_input("Adjusted High Price", value=152.0)
adjLow = st.number_input("Adjusted Low Price", value=148.0)
adjOpen = st.number_input("Adjusted Open Price", value=149.0)
adjVolume = st.number_input("Adjusted Volume", value=5000000)
divCash = st.number_input("Dividend Cash", value=0.5)
splitFactor = st.number_input("Split Factor", value=1.0)

# Prepare the input data
input_data = np.array([[
    close, high, low, open_price, volume,
    adjClose, adjHigh, adjLow, adjOpen, adjVolume,
    divCash, splitFactor
]])

# Make predictions when the "Predict" button is clicked
if st.button("Predict"):
    try:
        # Predict using the loaded model
        predicted_price = model.predict(input_data)
        st.success(f"The predicted stock price is: {predicted_price[0][0]:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Add optional forecasting
st.write("### Forecast Future Stock Prices")
future_days = st.slider("Select the number of future days to forecast:", min_value=1, max_value=30, value=7)

if st.button("Forecast"):
    try:
        # Forecast logic (e.g., extending data with predictions)
        forecast = []
        last_input = input_data

        for _ in range(future_days):
            prediction = model.predict(last_input)
            forecast.append(prediction[0][0])
            # Update the last input for next prediction
            last_input = np.roll(last_input, -1)
            last_input[0, -1] = prediction  # Replace with the new prediction

        # Display forecast results
        forecast_df = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(future_days)],
            "Predicted Price": forecast
        })
        st.write(forecast_df)
        st.line_chart(forecast_df.set_index("Day"))
    except Exception as e:
        st.error(f"Error in forecasting: {e}")
