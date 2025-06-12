import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import datetime
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="datarefresh")

st.title("📈 Bitcoin Price Predictor (BTC/USD)")

# Load BTC data
def load_data():
    df = yf.download("BTC-USD", period="1d", interval="1m")
    df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
    df.dropna(inplace=True)
    return df

df = load_data()

# Add indicators
def add_indicators(df):
    close = df['Close'].squeeze()
    df['rsi'] = ta.momentum.RSIIndicator(close=close).rsi()
    df['macd'] = ta.trend.MACD(close=close).macd()
    df['ema'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df['roc'] = ta.momentum.ROCIndicator(close=close).roc()
    df['bb_width'] = (
        ta.volatility.BollingerBands(close=close).bollinger_hband() -
        ta.volatility.BollingerBands(close=close).bollinger_lband()
    )
    df['target'] = df['Close'].shift(-3)  # Predict price 3 mins into the future
    df.dropna(inplace=True)
    return df

df = add_indicators(df)

# Features & Labels
features = ['rsi', 'macd', 'ema', 'roc', 'bb_width']
x = df[features]
y = df['target']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x[:-10], y[:-10])  # Leave last 10 for live prediction

# Make prediction on most recent data
latest = x[-1:]
predicted_price = model.predict(latest)[0]
actual_price = df['Close'].iloc[-1]
error_margin = abs(predicted_price - actual_price)

# Display predictions
st.subheader("📊 Live Prediction")
st.metric("Actual Price", f"${actual_price:,.2f}")
st.metric("Predicted Price (3min)", f"${predicted_price:,.2f}")
st.metric("Difference", f"${error_margin:,.2f}")

# Chart: Actual vs Predicted
st.subheader("📉 BTC Price vs Prediction (Recent)")
df['Predicted'] = np.append(model.predict(x[:-1]), [predicted_price])
chart_data = df[['Close', 'Predicted']].tail(50)
st.line_chart(chart_data)

st.caption("🚨 This bot predicts the BTC price 3 minutes ahead using live 1-minute indicators.")
