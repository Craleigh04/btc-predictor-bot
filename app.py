import streamlit as st
st.set_page_config(layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from streamlit_autorefresh import st_autorefresh
import os

st_autorefresh(interval=60 * 1000, key="refresh")
st.title("Bitcoin Momentum Analyzer Bot (BTC/USD)")
st.caption("Real-time BTC/USD forecast using technical indicators and Random Forest")

CACHE_FILE = "btc_data_cache.csv"

def load_btc_data():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            if 'Datetime' not in df.columns:
                raise ValueError("Cached file missing 'Datetime'")
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', utc=True)
            df = df[df['Datetime'] > pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)]
        except Exception as e:
            st.warning(f"Error loading cached data: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    recent = yf.download("BTC-USD", period="1d", interval="1m")
    if not recent.empty:
        recent.reset_index(inplace=True)
        recent.rename(columns={'index': 'Datetime', 'Date': 'Datetime', 'datetime': 'Datetime'}, inplace=True)
        recent['Datetime'] = pd.to_datetime(recent['Datetime'], errors='coerce', utc=True)
    else:
        recent = pd.DataFrame(columns=['Datetime'])

    combined = pd.concat([df, recent], ignore_index=True)

    if 'Datetime' not in combined.columns:
        st.error("Critical error: 'Datetime' column is missing from combined data.")
        return pd.DataFrame()

    combined['Datetime'] = pd.to_datetime(combined['Datetime'], errors='coerce', utc=True)
    combined = combined.dropna(subset=['Datetime'])
    combined = combined.drop_duplicates(subset='Datetime', keep='last').sort_values('Datetime').reset_index(drop=True)
    combined.to_csv(CACHE_FILE, index=False)
    return combined

# Load and verify data
df = load_btc_data()
if 'Close' in df.columns:
    df.rename(columns={'Close': 'Close_BTC-USD'}, inplace=True)

if df.empty or 'Close_BTC-USD' not in df.columns:
    st.error("Unable to retrieve BTC price data.")
    st.stop()

if len(df) < 50:
    st.warning("Not enough historical data to train the model. Please wait for more data to accumulate.")
    st.stop()

# Technical indicators
try:
    close_series = df['Close_BTC-USD']
    df['RSI'] = RSIIndicator(close=close_series).rsi()
    df['EMA'] = EMAIndicator(close=close_series, window=14).ema_indicator()
    df['MACD'] = MACD(close=close_series).macd()
    df['ROC'] = ROCIndicator(close=close_series).roc()
    df['BB_width'] = BollingerBands(close=close_series).bollinger_wband()
except Exception as e:
    st.error(f"Error calculating indicators: {e}")
    st.stop()

# Prediction
df['Target'] = close_series.shift(-3)
df = df.dropna().reset_index(drop=True)

features = ['Close_BTC-USD', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']
X = df[features]
y = df['Target']

if len(df) < 50:
    st.warning("Still not enough processed data to build model.")
    st.stop()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted'] = model.predict(X)

# Buy/Sell
df['Signal'] = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', ''))
buy_signals = df[df['Signal'] == 'Buy']
sell_signals = df[df['Signal'] == 'Sell']

# Live forecast
latest_input = df.iloc[-1][features].values.reshape(1, -1)
future_price = model.predict(latest_input)[0]
actual_price = close_series.iloc[-1]
predicted_time = df.iloc[-1]['Datetime'] + pd.Timedelta(minutes=3)

st.subheader("Live BTC Price Forecast")
col1, col2, col3 = st.columns(3)
col1.metric("Actual Price", f"${actual_price:,.2f}")
col2.metric("Predicted (3 min)", f"${future_price:,.2f}")
col3.metric("Time Predicted", predicted_time.strftime("%H:%M:%S"))

# Chart controls
st.subheader("Indicator Trend Visualization")
time_window = st.radio("Select time window:", ['1h', '6h', '24h'], horizontal=True)
show_signals = st.checkbox("Show Buy/Sell Signals", value=True)

if time_window == '1h':
    df_filtered = df[df['Datetime'] > df['Datetime'].max() - pd.Timedelta(hours=1)]
elif time_window == '6h':
    df_filtered = df[df['Datetime'] > df['Datetime'].max() - pd.Timedelta(hours=6)]
else:
    df_filtered = df.copy()

# Chart
available_indicators = ['Close_BTC-USD', 'EMA', 'RSI', 'MACD', 'ROC', 'BB_width', 'Predicted']
selected = st.multiselect("Select indicators to display:", available_indicators, default=['Close_BTC-USD', 'EMA', 'Predicted'])

if selected:
    plot_df = df_filtered[['Datetime'] + selected]
    melted = plot_df.melt(id_vars='Datetime', var_name='Metric', value_name='Value')

    fig = px.line(
        melted,
        x='Datetime',
        y='Value',
        color='Metric',
        title="BTC/USD Technical Indicators",
        hover_data={"Datetime": "|%Y-%m-%d %H:%M:%S", "Value": ":.2f"},
    )

    if show_signals:
        fig.add_trace(go.Scatter(
            x=buy_signals['Datetime'],
            y=buy_signals['Close_BTC-USD'],
            mode='markers',
            marker=dict(color='green', size=6, symbol='triangle-up', opacity=0.8),
            name='Buy Signal',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals['Datetime'],
            y=sell_signals['Close_BTC-USD'],
            mode='markers',
            marker=dict(color='red', size=6, symbol='triangle-down', opacity=0.8),
            name='Sell Signal',
            showlegend=True
        ))

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(title="Datetime", type="date", rangeslider_visible=True),
        yaxis_title="Value",
        margin=dict(l=30, r=30, t=40, b=30),
        dragmode="pan"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select indicators to display the chart.")
