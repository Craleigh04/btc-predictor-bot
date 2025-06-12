import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from streamlit_autorefresh import st_autorefresh

# 🔄 Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="datarefresh")

st.title("📈 Bitcoin Price Predictor (BTC/USD)")

# 📥 Load live BTC/USD data
def load_data():
    df = yf.download("BTC-USD", period="1d", interval="1m")
    df.dropna(inplace=True)
    return df

df = load_data()

# 🧠 Dynamically find the close column (e.g., Close_BTC-USD)
close_col = [col for col in df.columns if "Close" in col][0]

# 🧮 Add indicators
def add_indicators(df, close_col):
    close = df[close_col].squeeze()
    df['rsi'] = ta.momentum.RSIIndicator(close=close).rsi()
    df['macd'] = ta.trend.MACD(close=close).macd()
    df['ema'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df['roc'] = ta.momentum.ROCIndicator(close=close).roc()
    df['bb_width'] = (
        ta.volatility.BollingerBands(close=close).bollinger_hband() -
        ta.volatility.BollingerBands(close=close).bollinger_lband()
    )
    df['target'] = df[close_col].shift(-3)
    df.dropna(inplace=True)
    return df

df = add_indicators(df, close_col)

# 🧠 Features + labels
features = ['rsi', 'macd', 'ema', 'roc', 'bb_width']
x = df[features]
y = df['target']

# 🎯 Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x[:-10], y[:-10])  # Leave last 10 for prediction

# 📈 Predict next 3-minute price
latest = x[-1:]
predicted_price = model.predict(latest)[0]
actual_price = df[close_col].iloc[-1]
error_margin = abs(predicted_price - actual_price)

# 📊 Display results
st.subheader("📊 Live Prediction")
st.metric("Actual Price", f"${actual_price:,.2f}")
st.metric("Predicted Price (3min)", f"${predicted_price:,.2f}")
st.metric("Difference", f"${error_margin:,.2f}")

# 📉 Plot chart
st.subheader("📉 BTC Price vs Prediction")
df['Predicted'] = np.append(model.predict(x[:-1]), [predicted_price])
st.line_chart(df.loc[:, [close_col, "Predicted"]].tail(50))

st.caption("🚨 AI model forecasts BTC price 3 minutes ahead. Auto-refreshes every 60 seconds.")
