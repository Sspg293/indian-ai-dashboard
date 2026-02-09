
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Indian AI Market Dashboard", layout="centered")

st.title("📊 Indian AI Market Dashboard")
st.caption("Advanced AI Model + Interactive Charts")

def generate_signal(symbol):

    data = yf.download(symbol, start="2013-01-01")

    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(5).std()

    data.dropna(inplace=True)

    data['Target_1D'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['Target_5D'] = (data['Close'].shift(-5) > data['Close']).astype(int)

    data.dropna(inplace=True)

    features = ['MA20','MA50','RSI','Return','Volatility']

    X = data[features]
    y1 = data['Target_1D']
    y5 = data['Target_5D']

    split = int(len(data) * 0.8)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_latest = scaler.transform(X.iloc[-1:])

    model1 = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model5 = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model1.fit(X_train, y1[:split])
    model5.fit(X_train, y5[:split])

    prob1 = model1.predict_proba(X_latest)[0][1]
    prob5 = model5.predict_proba(X_latest)[0][1]

    price = data['Close'].iloc[-1]
    prev = data['Close'].iloc[-2]
    pct = ((price - prev) / prev) * 100

    def signal(prob):
        if prob > 0.60:
            return "BULLISH"
        elif prob < 0.40:
            return "BEARISH"
        else:
            return "NEUTRAL"

    return data, price, pct, signal(prob1), prob1, signal(prob5), prob5


def plot_chart(data, name):

    recent = data.tail(120)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent['Close'],
        name="Close"
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
        height=400,
        title=name
    )

    st.plotly_chart(fig, use_container_width=True)


assets = {
    "NIFTY 50": "^NSEI",
    "Gold ETF": "GOLDBEES.NS",
    "Silver ETF": "SILVERBEES.NS"
}

bullish_count = 0

for name, symbol in assets.items():

    data, price, pct, daily, prob1, weekly, prob5 = generate_signal(symbol)

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

    plot_chart(data, name)

    st.divider()

st.header("📈 Overall Market Sentiment")

if bullish_count >= 2:
    sentiment = "BULLISH"
    color = "green"
elif bullish_count == 1:
    sentiment = "NEUTRAL"
    color = "orange"
else:
    sentiment = "BEARISH"
    color = "red"

st.markdown(f"<h2 style='color:{color};'>{sentiment}</h2>", unsafe_allow_html=True)
