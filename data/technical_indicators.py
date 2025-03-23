import pandas as pd

# Function to generate technical indicators
def add_technical_indicators(data):
    # Moving Averages
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # RSI
    data['RSI'] = compute_rsi(data['Close'])

    # MACD
    data['MACD'] = compute_macd(data['Close'])

    # Daily Return
    data['Daily_Return'] = data['Close'].pct_change()

    # Volatility (10-day rolling standard deviation of daily returns)
    data['Volatility'] = data['Daily_Return'].rolling(window=10).std()

    # Price Change (Close - Open)
    data['Price_Change'] = data['Close'] - data['Open']

    # Drop NaN values created by rolling calculations
    data.dropna(inplace=True)
    return data

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal