import pandas as pd
import plotly.express as px
import streamlit as st


# Creates an interactive Plotly graph
def plot_interactive(data, predictions=None, forecast=None, stock_name="Stock"):
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

    # Create the base plot for historical data
    fig = px.line(data, x='Date', y='Close', title=f'{stock_name} Price Over Time',
                  labels={'Close': 'Price'},
                  color_discrete_sequence= ['#83C8FE']) #['#1E90FF']) #['blue'])  # Historical data in blue
    fig.update_traces(showlegend=True, name=f'{stock_name} Stock Price Over Time')  # Add legend with a custom name

    # Makes plot size better with Legend
    fig.update_layout(
        legend=dict(x=0.5, y=-0.2, orientation='h'),  # Position legend below the plot
        margin=dict(l=20, r=20, t=40, b=80),  # Adjust margins to accommodate the legend
        autosize=True  # Ensure the plot resizes properly
    )

    # Add predictions if available
    if predictions is not None:
        prediction_dates = pd.to_datetime(data['Date'][-len(predictions):], format='%d-%m-%Y')
        fig.add_scatter(x=prediction_dates, y=predictions, mode='lines',
                        name=f'{stock_name} Predicted Price', line=dict(color='orange'))  # Predictions in orange

    # Add forecast if available
    if forecast is not None:
        if 'ds' in forecast.columns:  # Prophet forecast
            forecast['ds'] = pd.to_datetime(forecast['ds'], format='%d-%m-%Y')
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                            name=f'{stock_name} Forecasted Price', line=dict(color='orange'))  # Forecast in orange
        elif 'Date' in forecast.columns:  # ARIMA forecast
            forecast['Date'] = pd.to_datetime(forecast['Date'], format='%d-%m-%Y')
            fig.add_scatter(x=forecast['Date'], y=forecast['Predicted Price'], mode='lines',
                            name=f'{stock_name} Forecasted Price', line=dict(color='orange'))  # Forecast in orange

    # Update the layout for better readability
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# Creates a separate prediction plot
def plot_predictions_only(predictions, title, stock_name="Stock"):
    fig = px.line(predictions, x='Date', y='Predicted Price', title=title,
                  labels={'Predicted Price': 'Price'},
                  color_discrete_sequence=['orange'])  # Predictions in orange

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)