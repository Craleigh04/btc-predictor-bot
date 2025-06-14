import streamlit as st
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

# 🔁 Refresh every 60s
st_autorefresh(interval=60 * 1000, key="refresh")

st.title("Bitcoin Momentum Analyzer Bot (BTC/USD)")
st.caption("Real-time BTC/USD forecast using technical indicators and Random Forest")

# 📦 Cache file
CACHE_FILE = "btc_data_cache.csv"

# 🧠 Load & backfill BTC data
def load_btc_data():
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df[df['Datetime'] > pd.Timestamp.now() - pd.Timedelta(days=7)]
    else:
        df = yf.download("BTC-USD", period="7d", interval="1m")
        df = df.reset_index()
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
        df.to_csv(CACHE_FILE, index=False)

    # Append latest 1d
    recent = yf.download("BTC-USD", period="1d", interval="1m").reset_index()
    recent.rename(columns={'Date': 'Datetime'}, inplace=True)
    recent['Datetime'] = pd.to_datetime(recent['Datetime'])
    df = pd.concat([df, recent], ignore_index=True)
    df = df.drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)
    df.to_csv(CACHE_FILE, index=False)
    return df

df = load_btc_data()

# 🧹 Validate
if df.empty or 'Close' not in df.columns:
    st.error("Unable to retrieve BTC price data.")
    st.stop()

# ➕ Indicators
df['Close'] = df['Close'].astype(float)
close_series = df['Close'].copy()
df['RSI'] = RSIIndicator(close=close_series).rsi()
df['EMA'] = EMAIndicator(close=close_series, window=14).ema_indicator()
df['MACD'] = MACD(close=close_series).macd()
df['ROC'] = ROCIndicator(close=close_series).roc()
df['BB_width'] = BollingerBands(close=close_series).bollinger_wband()
df['Target'] = close_series.shift(-3)

df.dropna(inplace=True)

# 🎯 Train model
features = ['Close', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']
X = df[features]
y = df['Target']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted'] = model.predict(X)

# 🔔 Buy/Sell markers (RSI)
df['Signal'] = np.where(df['RSI'] < 30, 'Buy',
                np.where(df['RSI'] > 70, 'Sell', ''))
buy_signals = df[df['Signal'] == 'Buy']
sell_signals = df[df['Signal'] == 'Sell']

# 🔮 Live forecast
latest_input = df.iloc[-1][features].values.reshape(1, -1)
future_price = model.predict(latest_input)[0]
actual_price = df['Close'].iloc[-1]
price_diff = future_price - actual_price
predicted_time = df['Datetime'].iloc[-1] + pd.Timedelta(minutes=3)

# 📊 Metrics
st.subheader("Live BTC Price Forecast")
col1, col2, col3 = st.columns(3)
col1.metric("Actual Price", f"${actual_price:,.2f}")
col2.metric("Predicted (3 min)", f"${future_price:,.2f}")
col3.metric("Predicted Time", predicted_time.strftime("%Y-%m-%d %H:%M:%S"))

# 📉 Time window
st.subheader("Indicator Trend Visualization")
time_window = st.radio("Select time window:", ["1h", "6h", "24h"], horizontal=True)
if time_window == "1h":
    df_plot = df[df['Datetime'] > df['Datetime'].max() - pd.Timedelta(hours=1)]
elif time_window == "6h":
    df_plot = df[df['Datetime'] > df['Datetime'].max() - pd.Timedelta(hours=6)]
else:
    df_plot = df

# ✅ Choose indicators
options = ['Close', 'EMA', 'RSI', 'MACD', 'ROC', 'BB_width', 'Predicted']
selected = st.multiselect("Select indicators to display:", options, default=['Close', 'EMA', 'Predicted'])

if selected:
    melted = df_plot[['Datetime'] + selected].melt(id_vars='Datetime', var_name='Metric', value_name='Value')

    fig = px.line(melted, x='Datetime', y='Value', color='Metric',
                  hover_data={"Datetime": True, "Value": ":.2f", "Metric": True},
                  title="BTC/USD Technical Indicators")

    fig.add_trace(go.Scatter(
        x=buy_signals['Datetime'],
        y=buy_signals['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=9, symbol='triangle-up')
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals['Datetime'],
        y=sell_signals['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=9, symbol='triangle-down')
    ))

    fig.update_layout(
        hovermode="x unified",
        dragmode="pan",
        xaxis=dict(rangeslider_visible=True, tickformat="%H:%M"),
        margin=dict(l=30, r=30, t=50, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select at least one indicator.")
