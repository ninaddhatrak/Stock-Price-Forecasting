# Code for ARIMA Model

import pandas as pd
#from statsmodels.tsa.arima.model import ARIMA
#from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import streamlit as st


# Function to find the best SARIMAX parameters
def find_best_sarimax_params(data, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    # Include exogenous variables
                    exog = data[['Open', 'High', 'Low', 'Volume']]
                    model = SARIMAX(data['Close'], order=(p, d, q), exog=exog)
                    model_fit = model.fit(disp=False)
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order


# Function to predict using SARIMAX with exogenous variables
def predict_arima(data, steps=60):
    # Find the best SARIMAX parameters
    p_range = range(0, 3)  # Test p values from 0 to 2
    d_range = range(0, 2)  # Test d values from 0 to 1
    q_range = range(0, 3)  # Test q values from 0 to 2
    best_order = find_best_sarimax_params(data, p_range, d_range, q_range)

    if best_order:
        st.write(f"Best SARIMAX Order: {best_order}")
        # Include exogenous variables
        exog = data[['Open', 'High', 'Low', 'Volume']]
        model = SARIMAX(data['Close'], order=best_order, exog=exog)
        model_fit = model.fit(disp=False)

        # Prepare future exogenous variables (use the last available values)
        future_exog = pd.DataFrame({
            'Open': [data['Open'].iloc[-1]] * steps,
            'High': [data['High'].iloc[-1]] * steps,
            'Low': [data['Low'].iloc[-1]] * steps,
            'Volume': [data['Volume'].iloc[-1]] * steps
        })

        # Forecast with exogenous variables
        forecast = model_fit.forecast(steps=steps, exog=future_exog)
        return forecast
    else:
        st.error("Failed to find suitable SARIMAX parameters.")
        return None