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

st_autorefresh(interval=60 * 1000, key="refresh")
st.set_page_config(layout="wide")

st.title("Bitcoin Momentum Analyzer Bot (BTC/USD)")
st.caption("Real-time BTC/USD forecast using technical indicators and Random Forest")

CACHE_FILE = "btc_data_cache.csv"

def load_btc_data():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', utc=True)
            df = df[df['Datetime'] > pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)]
        except Exception as e:
            st.warning(f"Error reading cached data: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    recent = yf.download("BTC-USD", period="1d", interval="1m")
    if not recent.empty:
        recent = recent.reset_index()
        recent.rename(columns={'index': 'Datetime', 'Date': 'Datetime', 'datetime': 'Datetime'}, inplace=True)
        recent['Datetime'] = pd.to_datetime(recent['Datetime'], errors='coerce', utc=True)
    else:
        recent = pd.DataFrame(columns=['Datetime'])

    combined = pd.concat([df, recent], ignore_index=True)
    if 'Datetime' not in combined.columns:
        st.error("Failed to retrieve 'Datetime' column.")
        return pd.DataFrame()

    combined['Datetime'] = pd.to_datetime(combined['Datetime'], errors='coerce', utc=True)
    combined = combined.dropna(subset=['Datetime'])
    combined = combined.drop_duplicates(subset='Datetime', keep='last').sort_values('Datetime').reset_index(drop=True)

    combined.to_csv(CACHE_FILE, index=False)
    return combined

# Load data
df = load_btc_data()
if df.empty or 'Close' not in df.columns:
    st.error("Unable to retrieve BTC price data.")
    st.stop()

df.rename(columns={'Close': 'Close_BTC-USD'}, inplace=True)

if len(df) < 50:
    st.warning("Not enough data to train the model. Please wait for more data to accumulate.")
    st.stop()

# Technical Indicators
try:
    close = df['Close_BTC-USD']
    df['RSI'] = RSIIndicator(close=close).rsi()
    df['EMA'] = EMAIndicator(close=close).ema_indicator()
    df['MACD'] = MACD(close=close).macd()
    df['ROC'] = ROCIndicator(close=close).roc()
    df['BB_width'] = BollingerBands(close=close).bollinger_wband()
except Exception as e:
    st.error(f"Indicator calculation error: {e}")
    st.stop()

df['Target'] = close.shift(-3)
df = df.dropna().reset_index(drop=True)

features = ['Close_BTC-USD', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']
X = df[features]
y = df['Target']

if len(X) < 50:
    st.warning("Still accumulating training data...")
    st.stop()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted'] = model.predict(X)

# Buy/Sell signals
df['Signal'] = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', ''))
buy_signals = df[df['Signal'] == 'Buy']
sell_signals = df[df['Signal'] == 'Sell']

# Prediction
latest = df.iloc[-1]
future_price = model.predict(latest[features].values.reshape(1, -1))[0]
predicted_time = latest['Datetime'] + pd.Timedelta(minutes=3)

# Metrics
st.subheader("Live BTC Price Forecast")
col1, col2, col3 = st.columns(3)
col1.metric("Actual Price", f"${latest['Close_BTC-USD']:.2f}")
col2.metric("Predicted (3 min)", f"${future_price:.2f}")
col3.metric("Time Predicted", predicted_time.strftime("%H:%M:%S"))

# Visualization
st.subheader("Indicator Trend Visualization")
time_range = st.radio("Select time window:", ['1h', '6h', '24h'], horizontal=True)
show_signals = st.checkbox("Show Buy/Sell Signals", value=True)

# Filter
now = df['Datetime'].max()
if time_range == '1h':
    df_filtered = df[df['Datetime'] > now - pd.Timedelta(hours=1)]
elif time_range == '6h':
    df_filtered = df[df['Datetime'] > now - pd.Timedelta(hours=6)]
else:
    df_filtered = df.copy()

# Indicator display
options = ['Close_BTC-USD', 'EMA', 'RSI', 'MACD', 'ROC', 'BB_width', 'Predicted']
selected = st.multiselect("Select indicators to display:", options, default=['Close_BTC-USD', 'EMA', 'Predicted'])

if selected:
    plot_df = df_filtered[['Datetime'] + selected]
    melted = plot_df.melt(id_vars='Datetime', var_name='Metric', value_name='Value')

    fig = px.line(
        melted,
        x='Datetime',
        y='Value',
        color='Metric',
        hover_data={'Datetime': "|%Y-%m-%d %H:%M:%S", 'Value': ':.2f'},
        title="BTC/USD Technical Indicators"
    )

    if show_signals:
        fig.add_trace(go.Scatter(
            x=buy_signals['Datetime'],
            y=buy_signals['Close_BTC-USD'],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=6),
            name='Buy Signal'
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals['Datetime'],
            y=sell_signals['Close_BTC-USD'],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=6),
            name='Sell Signal'
        ))

    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(rangeslider_visible=True, title="Datetime"),
        yaxis_title="Value",
        dragmode="pan",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Select at least one indicator to plot.")
