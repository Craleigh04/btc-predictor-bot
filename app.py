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

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="refresh")

st.title("Bitcoin Momentum Analyzer Bot (BTC/USD)")
st.caption("Real-time BTC/USD forecast using technical indicators and Random Forest")

CACHE_FILE = "btc_data_cache.csv"

def fetch_and_combine_btc_data():
    df_cache = pd.DataFrame()
    if os.path.exists(CACHE_FILE):
        try:
            df_cache = pd.read_csv(CACHE_FILE)
            if 'Datetime' in df_cache.columns:
                df_cache['Datetime'] = pd.to_datetime(df_cache['Datetime'], errors='coerce', utc=True)
        except Exception as e:
            st.warning(f"Cache file exists but couldn't be read: {e}")

    df_recent = yf.download("BTC-USD", period="1d", interval="1m")
    if not df_recent.empty:
        df_recent = df_recent.reset_index()
        df_recent.rename(columns={'index': 'Datetime', 'Date': 'Datetime', 'datetime': 'Datetime'}, inplace=True)
        df_recent['Datetime'] = pd.to_datetime(df_recent['Datetime'], errors='coerce', utc=True)
        combined = pd.concat([df_cache, df_recent], ignore_index=True)
    else:
        combined = df_cache

    if 'Datetime' not in combined.columns:
        st.error("Missing 'Datetime' in price data.")
        return pd.DataFrame()

    combined = combined.dropna(subset=['Datetime'])
    combined = combined.drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)
    combined.to_csv(CACHE_FILE, index=False)
    return combined

df = fetch_and_combine_btc_data()
if df.empty:
    st.error("Unable to retrieve BTC price data.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

if 'Close' in df.columns:
    df.rename(columns={'Close': 'Close_BTC-USD'}, inplace=True)

if 'Close_BTC-USD' not in df.columns:
    st.error("'Close_BTC-USD' column not found in data.")
    st.stop()

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

# Target column
df['Target'] = close_series.shift(-3)
df = df.dropna().reset_index(drop=True)

features = ['Close_BTC-USD', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']
X = df[features]
y = df['Target']

if X.empty or y.empty:
    st.error("Not enough data to train model.")
    st.stop()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted'] = model.predict(X)

# Buy/Sell Signals
signal_condition = np.where(df['RSI'] < 30, 'Buy', np.where(df['RSI'] > 70, 'Sell', ''))
df['Signal'] = signal_condition
buy_signals = df[df['Signal'] == 'Buy']
sell_signals = df[df['Signal'] == 'Sell']

# Latest Prediction
latest_input = df.iloc[-1][features].values.reshape(1, -1)
future_price = model.predict(latest_input)[0]
actual_price = close_series.iloc[-1]
predicted_time = df.iloc[-1]['Datetime'] + pd.Timedelta(minutes=3)

st.subheader("Live BTC Price Forecast")
col1, col2, col3 = st.columns(3)
col1.metric("Actual Price", f"${actual_price:,.2f}")
col2.metric("Predicted (3 min)", f"${future_price:,.2f}")
col3.metric("Time Predicted", predicted_time.strftime("%H:%M:%S"))

# Controls
st.subheader("Indicator Trend Visualization")
time_range = st.radio("Select time window:", ['1h', '6h', '24h'], horizontal=True)
show_signals = st.checkbox("Show Buy/Sell Signals", value=True)

if time_range == '1h':
    df_filtered = df[df['Datetime'] > df['Datetime'].max() - pd.Timedelta(hours=1)]
elif time_range == '6h':
    df_filtered = df[df['Datetime'] > df['Datetime'].max() - pd.Timedelta(hours=6)]
else:
    df_filtered = df.copy()

display_options = ['Close_BTC-USD', 'EMA', 'RSI', 'MACD', 'ROC', 'BB_width', 'Predicted']
selected = st.multiselect("Select indicators to display:", display_options, default=['Close_BTC-USD', 'EMA', 'Predicted'])

if selected:
    plot_df = df_filtered[['Datetime'] + selected]
    melted = plot_df.melt(id_vars='Datetime', var_name='Metric', value_name='Value')

    fig = px.line(
        melted,
        x='Datetime',
        y='Value',
        color='Metric',
        hover_data={"Datetime": "|%Y-%m-%d %H:%M:%S", "Value": ":.2f"},
        title="BTC/USD Technical Indicators"
    )

    if show_signals:
        fig.add_trace(go.Scatter(
            x=buy_signals['Datetime'],
            y=buy_signals['Close_BTC-USD'],
            mode='markers',
            marker=dict(color='green', size=6, symbol='triangle-up'),
            name='Buy Signal'
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals['Datetime'],
            y=sell_signals['Close_BTC-USD'],
            mode='markers',
            marker=dict(color='red', size=6, symbol='triangle-down'),
            name='Sell Signal'
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
    st.warning("Please select at least one indicator to display.")
