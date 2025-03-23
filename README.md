# 📈 Stock Price Forecasting

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Forecast stock prices**
The **Stock Price Forecasting** is a powerful, user-friendly tool built to help you forecast stock prices using advanced machine learning models. Whether you're an investor, trader, or data enthusiast, this app provides insights into future stock trends with just a few clicks.

---

## 🚀 Features

- **🌍 Multi-Market Support**: Analyze stocks from the US, Germany, Japan, India, China (SSE), China (SZE), Hong Kong, Australia, and Canada.
- **🤖 Multiple Models**: Choose from **LSTM**, **ARIMA**, or **Prophet** for predictions.
- **📊 Interactive Visualizations**: Beautiful, interactive charts powered by **Plotly**.
- **📉 Technical Indicators**: Includes moving averages, RSI, MACD, and more.
- **📥 Downloadable Results**: Export raw data and predictions as CSV files.
- **🐳 Docker Support**: Easy deployment with Docker for seamless setup.

---

## 🛠️ Why This Project?

Stock Price Forecasting is a challenging yet fascinating problem. This project demonstrates how **machine learning** and **time-series forecasting** can be applied to financial data to make informed decisions. It’s perfect for:
- **Learning**: Understand how machine learning models work with real-world data.
- **Research**: Experiment with different models and datasets.
- **Trading**: Gain insights into potential stock price movements.

---

## 🤖 Model Performance and Experience

In developing this app, I experimented with three different models: **LSTM**, **ARIMA**, and **Prophet**. Here’s a brief overview of my experience with each:

- **LSTM (Long Short-Term Memory)**: This model performed the best for stock price prediction. Its ability to capture temporal dependencies in sequential data made it highly effective for forecasting stock trends. The predictions were more accurate and aligned well with actual stock movements.

- **ARIMA (AutoRegressive Integrated Moving Average)**: While ARIMA is a classic time-series model, I found it less effective for stock price prediction. It struggled to capture the complexity and volatility of stock market data, often resulting in less accurate forecasts.

- **Prophet**: Developed by Facebook, Prophet is designed for time-series forecasting with a focus on simplicity. However, I found it less suitable for stock market data, as it often oversimplified trends and failed to account for sudden market fluctuations.

**Conclusion**: Based on my experience, **LSTM** is the more reliable model for stock price prediction, while ARIMA and Prophet were less effective for this specific use case.

---

## ⚠️ Disclaimer

**This project is for educational and research purposes only.**
The predictions generated by this app are based on historical data and machine learning models. However, the stock market is influenced by numerous unpredictable factors, including global events, economic conditions, and investor behavior. **The predictions provided by this app should not be taken as financial advice or used for making investment decisions.** Always consult with a qualified financial advisor before making any investment decisions.

---

## 🖥️ Demo

https://github.com/user-attachments/assets/357e66b5-1857-4caa-9b54-2af3cfacafdc

---

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- Docker (optional)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-forecasting.git
   cd Stock-Price-Forecasting

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run main.py

4. (Optional) Deploy with Docker:
   ```bash
   docker build -t stock-price-forecasting .
   docker run -p 8501:8501 stock-price-forecasting

---

## 🎯 Usage

1. **Select a Market**: Choose from US, EU, Japan, India, or China.
2. **Enter Stock Symbol**: For example, `AAPL` for Apple.
3. **Choose Date Range**: Select start and end dates for historical data.
4. **Pick a Model**: Choose between **LSTM**, **ARIMA**, or **Prophet**.
5. **Adjust Hyperparameters**: Tune model settings (if applicable).
6. **Run Predictions**: Click the button to generate forecasts.
7. **Explore Results**: View interactive plots and download predictions.

---

