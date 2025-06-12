import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="refresh")

st.title("🚀 Bitcoin Momentum Analyzer Bot (BTC/USD)")
st.caption("Live prediction bot using real-time indicators and Random Forest")

# Get data
df = yf.download("BTC-USD", period="1d", interval="1m").reset_index()
df.dropna(inplace=True)

# Identify 'Close' price
close_series = df['Close']

# Calculate indicators
df['RSI'] = RSIIndicator(close_series).rsi()
df['EMA'] = EMAIndicator(close_series, window=14).ema_indicator()
df['MACD'] = MACD(close_series).macd()
df['ROC'] = ROCIndicator(close_series).roc()
df['BB_width'] = BollingerBands(close_series).bollinger_wband()
df['Volume'] = df['Volume']

# Target variable (3-min ahead price)
df['Target'] = close_series.shift(-3)
df.dropna(inplace=True)

# Model Training
features = ['Close', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']
X = df[features]
y = df['Target']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted'] = pd.Series(model.predict(X), index=df.index)

# Latest prediction
latest_features = df.iloc[-1][features].values.reshape(1, -1)
future_price = model.predict(latest_features)[0]
actual_price = close_series.iloc[-1]
diff = future_price - actual_price

# Show metrics
st.subheader("📊 Live Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Actual", f"${actual_price:,.2f}")
col2.metric("Predicted (3min)", f"${future_price:,.2f}")
col3.metric("Difference", f"{diff:+.2f}")

# CHART TOGGLE OPTIONS
st.subheader("📈 BTC Chart (toggle indicators)")
indicator_opts = st.multiselect(
    "Choose indicators to show",
    options=['Close', 'EMA', 'RSI', 'MACD', 'ROC', 'BB_width', 'Predicted'],
    default=['Close', 'EMA', 'Predicted']
)

# Melt data for Altair
chart_data = df[['Datetime'] + indicator_opts].melt(id_vars='Datetime', var_name='Metric', value_name='Value')

highlight = alt.selection_multi(fields=['Metric'], bind='legend')

chart = alt.Chart(chart_data).mark_line().encode(
    x='Datetime:T',
    y='Value:Q',
    color='Metric:N',
    tooltip=['Datetime:T', 'Metric:N', 'Value:Q'],
    opacity=alt.condition(highlight, alt.value(1), alt.value(0.1))
).add_selection(
    highlight
).interactive()

st.altair_chart(chart, use_container_width=True)

st.caption("📉 Model updates every minute. Not financial advice.")

