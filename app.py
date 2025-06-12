import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from streamlit_autorefresh import st_autorefresh

# Set up auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="refresh")

# Title of the app
st.title("Bitcoin Price Predictor (BTC/USD)")

# Download 1-day of 1-minute BTC-USD data
data = yf.download("BTC-USD", period="1d", interval="1m")

# Handle column naming for Close price (account for potential multi-index or ticker suffix)
if isinstance(data.columns, pd.MultiIndex):
    # Flatten multi-index columns
    data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in data.columns]
# Identify the close price column name dynamically
close_cols = [c for c in data.columns if str(c).lower().startswith('close')]
if close_cols:
    close_col = close_cols[0]
else:
    st.error("Close price column not found in data.")
    st.stop()

close_series = data[close_col]

# Calculate technical indicators
rsi = RSIIndicator(close_series, window=14).rsi()
ema = EMAIndicator(close_series, window=14).ema_indicator()
macd_indicator = MACD(close_series, window_slow=26, window_fast=12, window_sign=9)
macd_line = macd_indicator.macd()            # MACD line
roc = ROCIndicator(close_series, window=12).roc()
bb_indicator = BollingerBands(close_series, window=20, window_dev=2)
bb_width = bb_indicator.bollinger_wband()    # Bollinger Band Width

# Prepare feature DataFrame
features_df = pd.DataFrame({
    'Close': close_series,
    'RSI': rsi,
    'EMA': ema,
    'MACD': macd_line,
    'ROC': roc,
    'BB_width': bb_width
})

# Prepare target variable (price 3 minutes into the future)
target = close_series.shift(-3)
# Combine features and target, drop rows with NaNs (initial periods and the last 3 rows without future target)
model_df = pd.concat([features_df, target.rename('Target')], axis=1).dropna()

# Train Random Forest model
X_train = model_df[['Close', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']]
y_train = model_df['Target']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Backfill predictions for historical data (to compare actual vs predicted)
train_preds = model.predict(X_train)
# Align predictions to the correct timestamps (shift forward by 3 minutes to match the target time)
pred_index = model_df.index + pd.Timedelta(minutes=3)
pred_series = pd.Series(train_preds, index=pred_index)
# Add the Predicted column to the main data frame (will align by index)
data['Predicted'] = pred_series

# Make prediction for the latest available data point (3 minutes into the future)
latest_features = features_df.iloc[-1][['Close', 'RSI', 'EMA', 'MACD', 'ROC', 'BB_width']].values.reshape(1, -1)
future_pred_price = model.predict(latest_features)[0]
latest_actual_price = float(close_series.iloc[-1])

# Display metrics: Actual Price, Predicted Price (3min ahead), and Difference
col1, col2, col3 = st.columns(3)
col1.metric("Actual Price", f"${latest_actual_price:,.2f}")
col2.metric("Predicted Price (3min)", f"${future_pred_price:,.2f}")
price_diff = future_pred_price - latest_actual_price
# Show difference with a plus/minus sign
col3.metric("Difference", f"{price_diff:+.2f}")

# Plot the actual vs. predicted price for the last 50 data points
st.subheader("Actual vs Predicted Price (Last 50 points)")
chart_df = pd.DataFrame({
    "Actual": close_series,
    "Predicted": data['Predicted']
})
st.line_chart(chart_df.tail(50))

# Disclaimer note
st.caption("Note: This is a basic predictive demo and not financial advice.")

