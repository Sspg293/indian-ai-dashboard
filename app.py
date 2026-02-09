import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Indian AI Market", layout="wide")

# -------------------------
# PREMIUM CSS
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

.asset-card {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
}

.price {
    font-size: 34px;
    font-weight: bold;
}

.up { color: #00ff9d; }
.down { color: #ff4b4b; }

.signal-bull {
    background: rgba(0,255,157,0.2);
    color: #00ff9d;
    padding: 6px 14px;
    border-radius: 25px;
    display: inline-block;
    margin-top: 10px;
}

.signal-bear {
    background: rgba(255,75,75,0.2);
    color: #ff4b4b;
    padding: 6px 14px;
    border-radius: 25px;
    display: inline-block;
    margin-top: 10px;
}

.signal-neutral {
    background: rgba(200,200,200,0.2);
    color: #dddddd;
    padding: 6px 14px;
    border-radius: 25px;
    display: inline-block;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Indian AI Market Dashboard")

# -------------------------
# MODEL
# -------------------------
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

    X = data[['MA20','MA50','RSI','Return']]
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


# -------------------------
# ASSETS
# -------------------------
assets = {
    "NIFTY 50": "^NSEI",
    "Gold ETF": "GOLDBEES.NS",
    "Silver ETF": "SILVERBEES.NS"
}

col1, col2, col3 = st.columns(3)

columns = [col1, col2, col3]

overall = 0

for i, (name, symbol) in enumerate(assets.items()):

    result = generate_signal(symbol)
    if result is None:
        continue

    data, price, pct, signal, prob = result

    if signal == "BULLISH":
        overall += 1
    elif signal == "BEARISH":
        overall -= 1

    arrow = "▲" if pct > 0 else "▼"
    color = "up" if pct > 0 else "down"

    badge_class = "signal-bull" if signal=="BULLISH" else \
                  "signal-bear" if signal=="BEARISH" else \
                  "signal-neutral"

    with columns[i]:
        st.markdown(f"""
        <div class="asset-card">
            <h3>{name}</h3>
            <div class="price">₹ {price:.2f}</div>
            <div class="{color}">{arrow} {pct:.2f}%</div>
            <div class="{badge_class}">{signal}</div>
        </div>
        """, unsafe_allow_html=True)


# -------------------------
# OVERALL SENTIMENT BANNER
# -------------------------
st.markdown("<br>", unsafe_allow_html=True)

if overall > 0:
    sentiment = "🚀 MARKET BULLISH"
elif overall < 0:
    sentiment = "⚠ MARKET BEARISH"
else:
    sentiment = "⚖ MARKET NEUTRAL"

st.markdown(f"""
<div class="main-card">
    <h2 style="text-align:center;">{sentiment}</h2>
</div>
""", unsafe_allow_html=True)


# -------------------------
# CHART SECTION
# -------------------------
st.subheader("Market Chart")

data = yf.download("^NSEI", period="6mo", auto_adjust=True)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
))

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)