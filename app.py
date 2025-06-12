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

# 🔄 Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="refresh")

st.title("🚀 Bitcoin Momentum Analyzer Bot (BTC/USD)")
st.caption("Live prediction bot using real-time indicators and Random Forest")

# 📥 Load BTC/USD data
df = yf.download("BTC-USD", period="1d", interval="1m")

# 🧾 Validate data
if df.empty or 'Close' not in df.columns:
    st.error("Failed to load BTC data. Try again later.")
    st.stop()

# 🧾 Reset index and guarantee 'Datetime' column
df = df.reset_index()

# Make sure a valid 'Datetime' column exists
if 'Datetime' not in df.columns:
    if 'index' in df.columns:
        df.rename(columns={'index': 'Datetime'}, inplace=True)
    elif 'Date' in df.columns:
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
    elif 'datetime' in df.columns:
        df.rename(columns={'datetime': 'Datetime'}, inplace=True)
    else:
        df['Datetime'] = pd.to_datetime(df.index)

df['Datetime'] = pd.to_datetime(df['Datetime'])
df.dropna(inplace=True)

# ✅ Ensure 1D Close series
try:
    close_series = pd.Series(df['Close'].values.flatten(), index=df.index)
except Exception as e:
    st.error(f"Failed to prepare Close series: {e}")
    st.stop()

# 🧮 Add indicators
try:
    df['RSI'] = RSIIndicator(close=close_series).rsi()
    df['EMA'] = EMAIndicator(close=close_series, window=14).ema_indicator()
    df['MACD'] = MACD(close=close_series).macd()
    df['ROC'] = ROCIndicator(close=close_series).roc()
    df['BB_width'] = BollingerBands(close=close_series).bollinger_wband()
except Exception as e:
    st.error(f"Indicator calculation error: {e}")
    st.stop()

# 🎯 Create prediction target
df['Target'] = close_series.shift(-3)
df.dropna(inplace=True)

# 🤖 Train model
features = ['Close', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']
X = df[features]
y = df['Target']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted'] = model.predict(X)

# 🔮 Make live prediction
latest_input = df.iloc[-1][features].values.reshape(1, -1)
future_price = model.predict(latest_input)[0]
actual_price = close_series.iloc[-1]
price_diff = future_price - actual_price

# 📊 Live metrics
st.subheader("📊 Live Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Actual", f"${actual_price:,.2f}")
col2.metric("Predicted (3min)", f"${future_price:,.2f}")
col3.metric("Difference", f"{price_diff:+.2f}")

# 📈 Chart with indicator toggles
st.subheader("📈 BTC Chart (Toggle Indicators)")
options = ['Close', 'EMA', 'RSI', 'MACD', 'ROC', 'BB_width', 'Predicted']
selected = st.multiselect("Select lines to display", options, default=['Close', 'EMA', 'Predicted'], key="indicator_selector")

# 🔐 Re-create 'Datetime' if it was dropped during transformations
if 'Datetime' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Datetime']):
    df['Datetime'] = pd.to_datetime(df.index)

# ✅ Plot if valid selection
if selected:
    existing = [col for col in selected if col in df.columns]
    if existing and 'Datetime' in df.columns:
        try:
            melted = df[['Datetime'] + existing].melt(id_vars='Datetime', var_name='Metric', value_name='Value')

            highlight = alt.selection_multi(fields=['Metric'], bind='legend')

            chart = alt.Chart(melted).mark_line().encode(
                x='Datetime:T',
                y='Value:Q',
                color='Metric:N',
                tooltip=['Datetime:T', 'Metric:N', 'Value:Q'],
                opacity=alt.condition(highlight, alt.value(1), alt.value(0.1))
            ).add_selection(
                highlight
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Chart generation failed: {e}")
    else:
        st.warning("Selected indicators are missing from the data.")
else:
    st.warning("Select at least one indicator to display the chart.")
