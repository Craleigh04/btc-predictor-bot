import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("📈 Bitcoin Momentum Analyzer Bot")

# Load data
def load_data():
    df = yf.download("BTC-USD", period="7d", interval="1h")
    df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
    df.dropna(inplace=True)
    return df

df = load_data()

# Find actual Close column name
close_col = [col for col in df.columns if 'Close' in col][0]

# Add technical indicators
def add_indicators(df, close_col):
    close = df[close_col].squeeze()
    df['rsi'] = ta.momentum.RSIIndicator(close=close).rsi()
    df['macd'] = ta.trend.MACD(close=close).macd()
    df['ema'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df['target'] = df[close_col].shift(-3) > df[close_col]
    df.dropna(inplace=True)
    return df

df = add_indicators(df, close_col)

# Train model
features = ['rsi', 'macd', 'ema']
x = df[features]
y = df['target']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x[:-10], y[:-10])

# Make predictions
latest = x[-1:]
pred = model.predict(latest)[0]
prob = model.predict_proba(latest)[0][int(pred)]

# Display results
st.subheader("📊 Latest Prediction")
if pred:
    st.success(f"Prediction: Price likely to go UP 📈 (Confidence: {prob:.2f})")
else:
    st.error(f"Prediction: Price likely to go DOWN 📉 (Confidence: {prob:.2f})")

# Plotting
st.subheader("📉 Price Chart with EMA")
df['Signal'] = model.predict(x)
st.line_chart(df[[close_col, 'ema']].dropna())

st.caption("Note: This is a basic predictive demo and not financial advice.")

