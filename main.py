# Stock Price Forcasting Streamlit Interface code

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from data.fetch_data import fetch_stock_name, fetch_stock_data, preprocess_data
from data.technical_indicators import add_technical_indicators
from models.lstm_model import run_lstm_predictions
from models.arima_model import predict_arima
from models.prophet_model import predict_prophet
from utils.visualization import plot_interactive, plot_predictions_only
import plotly.express as px

# Currency symbols for each market
MARKET_CURRENCY = {
    "US": "USD ($)",
    "Germany": "EUR (€)",
    "Japan": "JPY (¥)",
    "India": "INR (₹)",
    "Australia": "AUD (A$)",
    "Canada": "CAD (C$)",
    "China (SSE)": "CNY (¥)",
    "China (SZSE)": "CNY (¥)",
    "Hong Kong": "HKD (HK$)"
}

# Extensions for each market
MARKET_EXTENSION = {
    "US": "", # U.S. stocks, no extension needed
    "Germany": ".DE",  # German stocks listed on the XETRA exchange
    "Japan": ".T",  # Japanese stocks listed on the Tokyo Stock Exchange
    "India": ".NS",  # National Stock Exchange of India
    "Australia": ".AX",  # Australian Securities Exchange (ASX)
    "Canada": ".TO",  # Toronto Stock Exchange (TSX)
    "China (SSE)": ".SS",  # Shanghai Stock Exchange
    "China (SZSE)": ".SZ",  # Shenzhen Stock Exchange
    "Hong Kong": ".HK"  # Hong Kong Stock Exchange
}

# A few predefined stocks to select for each market
MARKET_STOCKS = {
    "US": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "Germany": ["SAP", "BMW", "VOW", "BAS", "SIE"],
    "Japan": ["7203", "9984", "9432", "6861", "8035"],
    "India": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"],
    "Australia": ["BHP", "CBA", "TLS", "WBC", "ANZ"],
    "Canada": ["TD", "BNS", "RY", "ENB", "SHOP"],
    "China (SSE)": ["601857", "601939", "600519", "601988", "601398"],
    "China (SZSE)": ["000002", "000651", "300750", "002594", "002714"],
    "Hong Kong": ["9988", "0700", "0005", "1299", "01810"]
}


def main():
    # To Display "Stock Price Forecasting"
    st.markdown(
        """
        <style>
        .big-font {
            font-size:50px !important;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<p class="big-font">Stock Price Forecasting</p>', unsafe_allow_html=True)

    # Code for Sidebar
    st.sidebar.header("Stock Selection")

    # Initialize session state for user input stock symbol
    if 'user_stock' not in st.session_state:
        st.session_state.user_stock = ""

    # Track previous selections
    if 'prev_market' not in st.session_state:
        st.session_state.prev_market = None
    if 'prev_stock' not in st.session_state:
        st.session_state.prev_stock = None

    # User inputs
    market = st.sidebar.selectbox("Select Stock Market", ["US", "Germany", "Japan", "India", "Australia", "Canada",
                                                          "China (SSE)", "China (SZSE)", "Hong Kong"])

    stock = st.sidebar.selectbox("Select Stock", MARKET_STOCKS[market])

    # Reset user input stock symbol, if market or stock selection changes
    if market != st.session_state.prev_market or stock != st.session_state.prev_stock:
        st.session_state.user_stock = ""
        st.session_state.prev_market = market
        st.session_state.prev_stock = stock

    # User input stock symbol input
    user_stock = st.sidebar.text_input("Or enter another stock symbol:", st.session_state.user_stock)

    # Update session state with user input stock symbol
    if user_stock != st.session_state.user_stock:
        st.session_state.user_stock = user_stock

    # Use user input stock symbol if provided
    if st.session_state.user_stock:
        stock = st.session_state.user_stock

    # Append the suffix to the stock symbol
    stock_with_suffix = stock + MARKET_EXTENSION[market]

    # Fetch the stock name dynamically
    stock_name = fetch_stock_name(stock_with_suffix)
    st.sidebar.write(f"**Selected Stock:** {stock_name} ({stock_with_suffix})")

    start_date = st.sidebar.date_input("Start Date:", datetime(2023, 3, 1))
    end_date = st.sidebar.date_input("End Date:", datetime(2025, 3, 14))
    model_choice = st.sidebar.selectbox("Choose model", ["LSTM", "ARIMA", "Prophet"])
    prediction_days = st.sidebar.slider("Select prediction days (1-28): ", 1, 28, 7)

    # User input in hyperparameter tuning for LSTM model
    if model_choice == "LSTM":
        with st.sidebar.expander("LSTM Hyperparameters", expanded=False):  # Collapsed by default
            st.subheader("LSTM Hyperparameters")
            units = st.number_input("Number of LSTM Units (50-200)", 50, 200, 100)
            epochs = st.number_input("Number of Epochs (10-100)", 10, 100, 50)
            batch_size = st.number_input("Batch Size (16-128)", 16, 128, 32)
            # can be st.slider if one wants

    # Fetch and display raw data
    if stock:
        stock_data = fetch_stock_data(stock_with_suffix, start_date, end_date)
        if stock_data is None or stock_data.empty:
            st.error("No data found for the given symbol and date range.")
            return

        st.subheader(f"{stock_name} - Stock Data (Last 10 Days) - Currency: {MARKET_CURRENCY[market]}")
        st.write(stock_data.tail(10))

        # Reset the index to include the Date column in the CSV
        stock_data_reset = stock_data.reset_index()

        # Debug: Print column names (if needed)
        # st.write("Column Names in stock_data_reset:", stock_data_reset.columns)

        # If column names are tuples, flatten them
        if isinstance(stock_data_reset.columns, pd.MultiIndex):
            stock_data_reset.columns = [' '.join(col).strip() for col in stock_data_reset.columns]

        st.download_button(
            label="Download Raw Data as CSV",
            data=stock_data_reset.to_csv(index=False).encode('utf-8'),
            file_name=f"{stock_name}_{stock_with_suffix}_raw_data.csv",
            mime="text/csv",
        )

        # Preprocess data
        processed_data = preprocess_data(stock_data)
        processed_data = add_technical_indicators(processed_data)

        # Stock Price Over Time plot
        st.subheader(f"{stock_name} ({stock_with_suffix}) - Stock Price Over Time")
        fig = px.line(stock_data_reset, x='Date', y=f'Close {stock_with_suffix}',
                      title=f"{stock_name} ({stock_with_suffix}) - Stock Price Over Time")
        fig.update_traces(showlegend=True, name=f'{stock_name} Stock Price Over Time')

        # Makes plot size better with Legend
        fig.update_layout(legend=dict(x=0.5, y=-0.2, orientation='h'),margin=dict(l=20, r=20, t=40, b=80),
                          autosize=True)
        st.plotly_chart(fig)

    # Code for generating the Forecasts
    if st.sidebar.button("Run Forecast"):
        with st.spinner("Generating Forecast..."):
            # If chosen LSTM Model
            if model_choice == "LSTM":
                st.subheader("LSTM Forecast")
                prediction_df, future_df, model = run_lstm_predictions(processed_data, lookback=60, units=units,
                                                                       epochs=epochs, batch_size=batch_size,
                                                                       prediction_days=prediction_days,
                                                                       end_date=end_date)
                st.write(future_df)
                st.download_button(
                    label="Download LSTM Forecast as CSV",
                    data=future_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{stock_with_suffix}_lstm_Forecast.csv",
                    mime="text/csv",
                )
                plot_interactive(processed_data, predictions=prediction_df['Predicted Price'], stock_name=stock_name)
                plot_predictions_only(future_df, title="LSTM Forecasted Price", stock_name=stock_name)

                # if you want to save the model
                #model.save(f"{stock_with_suffix}_lstm_model.h5")
                #st.success("LSTM model saved successfully!")

            # If chosen ARIMA Model
            elif model_choice == "ARIMA":
                st.subheader("ARIMA Forecast")
                forecast = predict_arima(processed_data, steps=prediction_days)
                if forecast is not None:
                    forecast_dates = pd.date_range(start=end_date + timedelta(days=1), periods=prediction_days)
                    forecast_dates = [date.strftime('%d-%m-%Y') for date in forecast_dates]
                    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': forecast})
                    st.write(forecast_df)
                    st.download_button(
                        label="Download ARIMA Forecast as CSV",
                        data=forecast_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"{stock_with_suffix}_arima_Forecast.csv",
                        mime="text/csv",
                    )
                    plot_interactive(processed_data, forecast=forecast_df, stock_name=stock_name)
                    plot_predictions_only(forecast_df, title="ARIMA Future Forecast", stock_name=stock_name)

            # If chosen Prophet Model
            elif model_choice == "Prophet":
                st.subheader("Prophet Forecast")
                forecast = predict_prophet(processed_data, steps=prediction_days)
                if forecast is not None:
                    forecast = forecast.tail(prediction_days)
                    st.write(forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price'}))
                    st.download_button(
                        label="Download Prophet Forecast as CSV",
                        data=forecast.to_csv(index=False).encode('utf-8'),
                        file_name=f"{stock_with_suffix}_prophet_Forecast.csv",
                        mime="text/csv",
                    )
                    plot_interactive(processed_data, forecast=forecast, stock_name=stock_name)
                    plot_predictions_only(forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price'}),
                                          title="Prophet Future Forecast", stock_name=stock_name)

    # Information about models
    st.sidebar.markdown("---")
    st.sidebar.subheader("Models")
    st.sidebar.write("""
    - **LSTM**: Long Short-Term Memory model for time series forecasting.
    - **ARIMA**: AutoRegressive Integrated Moving Average model.
    - **Prophet**: Facebook's forecasting model.
    """)

if __name__ == "__main__":
    main()