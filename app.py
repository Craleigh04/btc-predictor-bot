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

st.title("Bitcoin Momentum Analyzer Bot (BTC/USD)")
st.caption("Real-time BTC/USD forecast using technical indicators and Random Forest")

# Download BTC/USD data
df = yf.download("BTC-USD", period="1d", interval="1m")

# Flatten MultiIndex columns (e.g., from ('Close', 'BTC-USD') to 'Close_BTC-USD')
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Verify required columns
if df.empty or 'Close_BTC-USD' not in df.columns:
    st.error("Unable to retrieve BTC price data.")
    st.stop()

# Reset index and standardize 'Datetime'
df = df.reset_index()
df.rename(columns={'index': 'Datetime', 'Date': 'Datetime', 'datetime': 'Datetime'}, inplace=True)
if 'Datetime' not in df.columns:
    df['Datetime'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='min')
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Prepare Close series
try:
    close_series = pd.Series(df['Close_BTC-USD'].values.flatten(), index=df.index)
except Exception as e:
    st.error(f"Error preparing price series: {e}")
    st.stop()

# Compute indicators
try:
    df['RSI'] = RSIIndicator(close=close_series).rsi()
    df['EMA'] = EMAIndicator(close=close_series, window=14).ema_indicator()
    df['MACD'] = MACD(close=close_series).macd()
    df['ROC'] = ROCIndicator(close=close_series).roc()
    df['BB_width'] = BollingerBands(close=close_series).bollinger_wband()
except Exception as e:
    st.error(f"Indicator calculation error: {e}")
    st.stop()

# Target variable
df['Target'] = close_series.shift(-3)
df = df.dropna().reset_index(drop=True)

# Restore Datetime column if dropped
if 'Datetime' not in df.columns or df['Datetime'].isnull().any():
    df['Datetime'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='min')
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Train model
features = ['Close_BTC-USD', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']
X = df[features]
y = df['Target']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted'] = model.predict(X)

# Live forecast
latest_input = df.iloc[-1][features].values.reshape(1, -1)
future_price = model.predict(latest_input)[0]
actual_price = close_series.iloc[-1]
price_diff = future_price - actual_price

# Display results
st.subheader("Live BTC Price Forecast")
col1, col2, col3 = st.columns(3)
col1.metric("Actual Price", f"${actual_price:,.2f}")
col2.metric("Predicted (3 min)", f"${future_price:,.2f}")
col3.metric("Difference", f"{price_diff:+.2f}")

# Chart toggles
st.subheader("Indicator Trend Visualization")
options = ['Close_BTC-USD', 'EMA', 'RSI', 'MACD', 'ROC', 'BB_width', 'Predicted']
selected = st.multiselect("Select indicators to display:", options, default=['Close_BTC-USD', 'EMA', 'Predicted'], key="chart_selector")

# Recheck Datetime before plotting
if 'Datetime' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Datetime']):
    df['Datetime'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='min')
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Plot chart
if selected:
    existing = [col for col in selected if col in df.columns]
    if existing:
        try:
            melted = df[['Datetime'] + existing].copy().melt(id_vars='Datetime', var_name='Metric', value_name='Value')

            highlight = alt.selection_multi(fields=['Metric'], bind='legend')

            chart = alt.Chart(melted).mark_line().encode(
                x='Datetime:T',
                y='Value:Q',
                color='Metric:N',
                tooltip=['Datetime:T', 'Metric:N', 'Value:Q'],
                opacity=alt.condition(highlight, alt.value(1), alt.value(0.15))
            ).add_selection(
                highlight
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Chart generation failed: {e}")
            st.write("Debug: DataFrame Columns", df.columns.tolist())
            st.write("Debug: DataFrame Sample", df.head())
    else:
        st.warning("Selected columns are not available in the current dataset.")
else:
    st.warning("Please select at least one indicator to display the graph.")
