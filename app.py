
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import datetime
import pytz

st.set_page_config(page_title="Indian AI Market Dashboard", layout="centered")

st.title("📊 Indian AI Market Dashboard")
st.caption("Multi-Asset AI | Daily + Weekly | 7-Day Projection")

# Auto refresh every 10 minutes
st_autorefresh(interval=600000, key="refresh")

def is_after_market_close():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(ist)
    market_close = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return now >= market_close

@st.cache_data(ttl=86400 if is_after_market_close() else 600)
def generate_signal(symbol):

    data = yf.download(symbol, start="2015-01-01", auto_adjust=True)

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(5).std()

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    if len(data) < 200:
        return None

    data['Target_1D'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['Target_5D'] = (data['Close'].shift(-5) > data['Close']).astype(int)
    data.dropna(inplace=True)

    features = ['MA20','MA50','RSI','Return','Volatility']

    X = data[features]
    y1 = data['Target_1D']
    y5 = data['Target_5D']

    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_latest = X.iloc[[-1]]

    model1 = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model5 = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model1.fit(X_train, y1.iloc[:split])
    model5.fit(X_train, y5.iloc[:split])

    prob1 = model1.predict_proba(X_latest)[0][1]
    prob5 = model5.predict_proba(X_latest)[0][1]

    price = float(data['Close'].iloc[-1])
    prev = float(data['Close'].iloc[-2])
    pct = ((price - prev) / prev) * 100

    def signal(prob):
        if prob > 0.60:
            return "BULLISH"
        elif prob < 0.40:
            return "BEARISH"
        else:
            return "NEUTRAL"

    return data, price, pct, signal(prob1), prob1, signal(prob5), prob5

def forecast_7_days(data, prob):
    last_price = float(data['Close'].iloc[-1])
    avg_vol = data['Return'].std()
    bias = (prob - 0.5) * 2

    projected_prices = []
    price = last_price

    for _ in range(7):
        expected_move = bias * avg_vol
        price = price * (1 + expected_move)
        projected_prices.append(price)

    return projected_prices

def plot_chart(data, name, projection):

    recent = data.tail(120)
    last_date = recent.index[-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MA20'], name="MA20"))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MA50'], name="MA50"))

    future_dates = pd.date_range(start=last_date, periods=8, freq='B')[1:]

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=projection,
        name="7-Day Projection",
        line=dict(dash="dash")
    ))

    fig.update_layout(template="plotly_dark", height=450, title=name)

    st.plotly_chart(fig, use_container_width=True)

assets = {
    "NIFTY 50": "^NSEI",
    "Nippon Gold ETF": "GOLDBEES.NS",
    "Nippon Silver ETF": "SILVERBEES.NS",
    "HDFC Gold ETF": "HDFCGOLD.NS",
    "HDFC Silver ETF": "HDFCSILVER.NS"
}

bullish_count = 0

for name, symbol in assets.items():

    result = generate_signal(symbol)

    if result is None:
        st.warning(f"Not enough data for {name}")
        continue

    data, price, pct, daily, prob1, weekly, prob5 = result
    projection = forecast_7_days(data, prob1)

    if weekly == "BULLISH":
        bullish_count += 1

    color = "green" if pct > 0 else "red"
    arrow = "▲" if pct > 0 else "▼"

    st.subheader(name)
    st.markdown(f"### ₹ {price:.2f}")
    st.markdown(f"<h4 style='color:{color};'>{arrow} {pct:.2f}%</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("📊 Daily Outlook:", daily)
        st.progress(float(prob1))

    with col2:
        st.write("📅 Weekly Outlook:", weekly)
        st.progress(float(prob5))

    plot_chart(data, name, projection)
    st.divider()

st.header("📈 Overall Market Sentiment")

if bullish_count >= 3:
    st.markdown("<h2 style='color:green;'>STRONG BULLISH</h2>", unsafe_allow_html=True)
elif bullish_count >= 2:
    st.markdown("<h2 style='color:orange;'>MODERATE BULLISH</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='color:red;'>WEAK / BEARISH</h2>", unsafe_allow_html=True)
