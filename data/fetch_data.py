import yfinance as yf
import pandas as pd
import streamlit as st

# Function to fetch stock name using yfinance
def fetch_stock_name(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        return stock_info.get('longName', 'Unknown Stock')
    except Exception as e:
        st.error(f"Error fetching stock name: {e}")
        return "Unknown Stock"

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("No data found for the given symbol and date range.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to preprocess data
def preprocess_data(stock_data):
    # Flatten the stock_data DataFrame if it has a MultiIndex
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)  # Flatten columns
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.strftime('%d-%m-%Y')
    return stock_data