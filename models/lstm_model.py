# Code for LSTM Model

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

def run_lstm_predictions(data, lookback=60, units=100, epochs=50, batch_size=32, prediction_days=7, end_date=None):
    # Prepare data
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'Daily_Return', 'Volatility', 'Price_Change']
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(data[features])
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i - lookback:i])
        y.append(scaled_target[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    # Train model
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    # Generate predictions
    predictions = model.predict(X)
    predictions = target_scaler.inverse_transform(predictions)
    prediction_dates = data['Date'][lookback:]
    prediction_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': predictions.flatten()})

    # Generate future predictions
    future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=prediction_days)
    future_dates = [date.strftime('%d-%m-%Y') for date in future_dates]
    future_predictions = predictions[-prediction_days:]
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})

    return prediction_df, future_df, model