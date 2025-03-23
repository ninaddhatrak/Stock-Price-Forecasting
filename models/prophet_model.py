# Code for Prophet Model

import pandas as pd
from prophet import Prophet
#from prophet.diagnostics import cross_validation, performance_metrics

# Function to predict using Prophet with additional regressors
def predict_prophet(data, steps=60):
    # Prepare data for Prophet
    prophet_data = data[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%d-%m-%Y')

    # Initialize Prophet model with additional regressors
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_regressor('Open')
    model.add_regressor('High')
    model.add_regressor('Low')
    model.add_regressor('Volume')

    # Fit the model
    model.fit(prophet_data)

    # Create future dataframe with additional regressors
    future = model.make_future_dataframe(periods=steps)
    future['Open'] = data['Open'].iloc[-1]  # Use the last available value
    future['High'] = data['High'].iloc[-1]
    future['Low'] = data['Low'].iloc[-1]
    future['Volume'] = data['Volume'].iloc[-1]

    # Make predictions
    forecast = model.predict(future)
    forecast['ds'] = forecast['ds'].dt.strftime('%d-%m-%Y')
    return forecast[['ds', 'yhat']]