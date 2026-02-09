import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Indian AI Market Dashboard", layout="wide")

# -----------------------------
# CUSTOM DARK CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.price {
    font-size: 32px;
    font-weight: bold;
}
.green { color: #00ff88; }
.red { color: #ff4b4b; }
.badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}
.bull { background-color: #003d2e; color: #00ff88; }
.bear { background-color: #3d0000; color: #ff4b4b; }
.neutral { background-color: #333333; color: #cccccc; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Indian AI Market Dashboard")

# -----------------------------
# MODEL FUNCTION
# -----------------------------
@st.cache_data
def generate_signal(symbol):

    data = yf.download(symbol, start="2016-01-01", auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[['Open','High','Low','Close','Volume']]
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    if len(data) < 200:
        return None

    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)

    features = ['MA20','MA50','RSI','Return']
    X = data[features]
    y = data['Target']

    split = int(len(data) * 0.8)

    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X.iloc[:split], y.iloc[:split])

    prob = model.predict_proba(X.iloc[[-1]])[0][1]

    price = float(data['Close'].iloc[-1])
    prev = float(data['Close'].iloc[-2])
    pct = ((price - prev) / prev) * 100

    if prob > 0.6:
        signal = "BULLISH"
    elif prob < 0.4:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return data, price, pct, signal, prob


# -----------------------------
# CANDLESTICK CHART
# -----------------------------
def plot_chart(data, name):

    recent = data.tail(100)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent['Open'],
        high=recent['High'],
        low=recent['Low'],
        close=recent['Close'],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent['MA20'],
        name="MA20"
    ))

    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent['MA50'],
        name="MA50"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# ASSETS
# -----------------------------
assets = {
    "NIFTY 50": "^NSEI",
    "Gold ETF": "GOLDBEES.NS",
    "Silver ETF": "SILVERBEES.NS"
}

overall_score = 0

for name, symbol in assets.items():

    result = generate_signal(symbol)

    if result is None:
        st.warning(f"Not enough data for {name}")
        continue

    data, price, pct, signal, prob = result

    if signal == "BULLISH":
        overall_score += 1
    elif signal == "BEARISH":
        overall_score -= 1

    color = "green" if pct > 0 else "red"
    arrow = "▲" if pct > 0 else "▼"

    badge_class = "bull" if signal=="BULLISH" else "bear" if signal=="BEARISH" else "neutral"

    st.markdown(f"""
    <div class="card">
        <h3>{name}</h3>
        <div class="price">₹ {price:.2f}</div>
        <div class="{color}">{arrow} {pct:.2f}%</div>
        <div class="badge {badge_class}">{signal}</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(prob))

    plot_chart(data, name)

# -----------------------------
# OVERALL SENTIMENT
# -----------------------------
st.markdown("---")

if overall_score > 0:
    sentiment = "BULLISH MARKET"
    style = "green"
elif overall_score < 0:
    sentiment = "BEARISH MARKET"
    style = "red"
else:
    sentiment = "NEUTRAL MARKET"
    style = "neutral"

st.markdown(f"<h2 class='{style}'>{sentiment}</h2>", unsafe_allow_html=True)