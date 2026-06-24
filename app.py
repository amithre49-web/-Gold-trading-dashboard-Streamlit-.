"""
Gold Trading Dashboard — Beginner-friendly multi-timeframe strategy
- Data: Yahoo Finance (GC=F by default)
- Timeframes: 15m entries, 1H and 4H trend confirmation
- Entry: 15m EMA(8) crosses above EMA(21) AND 1H & 4H are bullish (EMA50>EMA200)
- Filters: RSI, minimum volume
- Stops/TP: ATR-based (stop = close - atr * stop_atr_mult for longs)
- Position sizing: risk_percent of equity
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Gold Beginner Trading Dashboard")

# ---------------- Helpers / indicators ----------------
@st.cache_data(ttl=120)
def download_15m(ticker="GC=F", period="60d"):
    df = yf.download(tickers=ticker, period=period, interval="15m", progress=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    return df[["Open","High","Low","Close","Volume"]]

def resample_ohlcv(df, minutes):
    if minutes == 15:
        return df.copy()
    rule = f"{minutes}T"
    agg = df.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    return agg.dropna()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def find_entries(df15, df1h, df4h, params):
    f = params
    df = df15.copy()
    df["EMA_fast"] = ema(df["Close"], f["ema_fast"])
    df["EMA_slow"] = ema(df["Close"], f["ema_slow"])
    df["EMA_50"] = ema(df["Close"], 50)
    df["EMA_200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"], f["rsi_period"])
    df["ATR"] = atr(df, f["atr_period"])

    h = df1h.copy()
    h["EMA_50"] = ema(h["Close"], 50)
    h["EMA_200"] = ema(h["Close"], 200)

    q = df4h.copy()
    q["EMA_50"] = ema(q["Close"], 50)
    q["EMA_200"] = ema(q["Close"], 200)

    h_trend = (h["EMA_50"] > h["EMA_200"]).reindex(df.index, method="ffill").fillna(False)
    q_trend = (q["EMA_50"] > q["EMA_200"]).reindex(df.index, method="ffill").fillna(False)

    prev_diff = (df["EMA_fast"].shift(1) - df["EMA_slow"].shift(1))
    curr_diff = (df["EMA_fast"] - df["EMA_slow"])
    cross_up = (prev_diff <= 0) & (curr_diff > 0)

    candidates = df.loc[cross_up].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["Datetime","Close","EMA_fast","EMA_slow","RSI","ATR","1H_trend","4H_trend","Stop","TP","Risk_per_lot","Position_size"])

    candidates["1H_trend"] = h_trend.loc[candidates.index]
    candidates["4H_trend"] = q_trend.loc[candidates.index]

    def pass_filters(row):
        if not (row["1H_trend"] and row["4H_trend"]):
            return False
        if row["RSI"] > f["rsi_max"]:
            return False
        if f["min_volume"] > 0 and row["Volume"] < f["min_volume"]:
            return False
        return True

    candidates["pass"] = candidates.apply(pass_filters, axis=1)
    candidates = candidates[candidates["pass"]]

    results = []
    equity = f["account_equity"]
    risk_pct = f["risk_percent"] / 100.0
    for idx, row in candidates.iterrows():
        close = row["Close"]
        atr_val = row["ATR"] if not np.isnan(row["ATR"]) else 0
        stop = close - atr_val * f["stop_atr_mult"]
        tp = close + atr_val * f["tp_atr_mult"]
        risk_per_unit = close - stop
        if risk_per_unit <= 0 or atr_val == 0:
            pos_size = np.nan
            risk_amount = np.nan
        else:
            risk_amount = equity * risk_pct
            pos_size = risk_amount / risk_per_unit
        results.append({
            "Datetime": idx,
            "Close": close,
            "EMA_fast": row["EMA_fast"],
            "EMA_slow": row["EMA_slow"],
            "RSI": row["RSI"],
            "ATR": atr_val,
            "1H_trend": row["1H_trend"],
            "4H_trend": row["4H_trend"],
            "Stop": stop,
            "TP": tp,
            "Risk_amount": risk_amount if not np.isnan(risk_amount) else None,
            "Position_size_units": pos_size if not np.isnan(pos_size) else None
        })
    return pd.DataFrame(results).set_index("Datetime")

# ---------------- Sidebar: user inputs ----------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="GC=F")
    period = st.selectbox("Download period (15m)", ["30d","60d","90d"], index=1)
    ema_fast = st.number_input("15m EMA fast (entry)", min_value=2, max_value=50, value=8)
    ema_slow = st.number_input("15m EMA slow (entry)", min_value=2, max_value=100, value=21)
    rsi_period = st.number_input("RSI period", min_value=2, max_value=50, value=14)
    rsi_max = st.number_input("Max RSI to allow entry (smaller = more conservative)", min_value=30, max_value=90, value=60)
    atr_period = st.number_input("ATR period", min_value=5, max_value=50, value=14)
    stop_atr_mult = st.number_input("Stop = ATR ×", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    tp_atr_mult = st.number_input("Take-profit = ATR ×", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    min_volume = st.number_input("Min volume (0 = ignore)", min_value=0, value=0)
    account_equity = st.number_input("Account equity (USD)", min_value=100.0, value=1000.0, step=100.0)
    risk_percent = st.number_input("Risk percent per trade (%)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    refresh = st.button("Refresh data / find signals")

params = {
    "ema_fast": int(ema_fast),
    "ema_slow": int(ema_slow),
    "rsi_period": int(rsi_period),
    "rsi_max": float(rsi_max),
    "atr_period": int(atr_period),
    "stop_atr_mult": float(stop_atr_mult),
    "tp_atr_mult": float(tp_atr_mult),
    "min_volume": float(min_volume),
    "account_equity": float(account_equity),
    "risk_percent": float(risk_percent)
}

st.title("Gold Trading Dashboard — Beginner Setup")
st.markdown("This dashboard implements a conservative, easy-to-understand approach: trade with the longer-term trend (4H+1H) and take 15m EMA cross entries with ATR-based stops.")

# ---------------- Data download ----------------n
df15 = download_15m(ticker=ticker, period=period)
if df15.empty:
    st.error("No data returned from Yahoo Finance. Try a different ticker (GC=F) or shorter period.")
    st.stop()

df15 = df15.dropna()
df1h = resample_ohlcv(df15, 60)
df4h = resample_ohlcv(df15, 240)

# compute EMAs / RSI / ATR for plotting
for d in (df15, df1h, df4h):
    d["EMA_fast"] = ema(d["Close"], ema_fast)
    d["EMA_slow"] = ema(d["Close"], ema_slow)
    d["RSI"] = rsi(d["Close"], rsi_period)
    d["ATR"] = atr(d, params["atr_period"])

# Find entry signals on 15m confirmed by 1H & 4H
rsi_max = None if rsi_max == 0 else rsi_max
signals = find_entries(df15, df1h, df4h, params)

# ---------- Layout: top summary ----------
col1, col2, col3 = st.columns(3)
col1.metric("Latest 15m Close", f"{df15['Close'].iloc[-1]:.4f}")
col2.metric("Latest 1H Close", f"{df1h['Close'].iloc[-1]:.4f}")
col3.metric("Latest 4H Close", f"{df4h['Close'].iloc[-1]:.4f}")

trend1 = "Bull" if df1h["EMA_fast"].iloc[-1] > df1h["EMA_slow"].iloc[-1] else "Bear"
trend4 = "Bull" if df4h["EMA_fast"].iloc[-1] > df4h["EMA_slow"].iloc[-1] else "Bear"
st.markdown(f"1H trend: **{trend1}** (EMA {ema_fast} vs {ema_slow}) • 4H trend: **{trend4}**")

# ---------- Plots ----------
st.subheader("15-minute chart (with entry signals)")
# ensure EMAs/RSI present on 15m passed to plotting function
df15["EMA_fast"] = ema(df15["Close"], ema_fast)
df15["EMA_slow"] = ema(df15["Close"], ema_slow)
df15["RSI"] = rsi(df15["Close"], rsi_period)

def plot_15m_with_signals(df15, signals, ema_fast, ema_slow, show_volume=True):
    rows = 3 if show_volume else 2
    row_heights = [0.6, 0.25, 0.15] if show_volume else [0.75, 0.25]
    specs = [[{"type":"xy"}],[{"type":"xy"}],[{"type":"xy"}]] if show_volume else [[{"type":"xy"}],[{"type":"xy"}]]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights, specs=specs)

    fig.add_trace(
        go.Candlestick(x=df15.index, open=df15["Open"], high=df15["High"], low=df15["Low"], close=df15["Close"], name="Price"),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df15.index, y=df15["EMA_fast"], line=dict(color="orange", width=1), name=f"EMA {ema_fast}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df15.index, y=df15["EMA_slow"], line=dict(color="purple", width=1), name=f"EMA {ema_slow}"), row=1, col=1)

    if show_volume:
        colors = np.where(df15["Close"] >= df15["Open"], "green", "red")
        fig.add_trace(go.Bar(x=df15.index, y=df15["Volume"], marker_color=colors, name="Volume"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df15.index, y=df15["RSI"], line=dict(color="blue", width=1), name=f"RSI ({rsi_period})"), row=rows, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=rows, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=rows, col=1)

    if not signals.empty:
        fig.add_trace(go.Scatter(x=signals.index, y=signals["Close"], mode="markers", marker=dict(symbol="triangle-up", color="lime", size=12), name="Entry Signal (15m)"), row=1, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(height=900, margin=dict(l=10, r=10, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
    else:
        fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig

fig = plot_15m_with_signals(df15, signals, ema_fast, ema_slow, show_volume=True)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show 1-hour chart"):
    fig1h = make_subplots(rows=1, cols=1, specs=[[{"type":"xy"}]])
    fig1h.add_trace(go.Candlestick(x=df1h.index, open=df1h["Open"], high=df1h["High"], low=df1h["Low"], close=df1h["Close"], name="1H Price"))
    fig1h.add_trace(go.Scatter(x=df1h.index, y=df1h["EMA_fast"], line=dict(color="orange", width=1), name=f"EMA {ema_fast}"))
    fig1h.add_trace(go.Scatter(x=df1h.index, y=df1h["EMA_slow"], line=dict(color="purple", width=1), name=f"EMA {ema_slow}"))
    fig1h.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig1h, use_container_width=True)

with st.expander("Show 4-hour chart"):
    fig4h = make_subplots(rows=1, cols=1, specs=[[{"type":"xy"}]])
    fig4h.add_trace(go.Candlestick(x=df4h.index, open=df4h["Open"], high=df4h["High"], low=df4h["Low"], close=df4h["Close"], name="4H Price"))
    fig4h.add_trace(go.Scatter(x=df4h.index, y=df4h["EMA_fast"], line=dict(color="orange", width=1), name=f"EMA {ema_fast}"))
    fig4h.add_trace(go.Scatter(x=df4h.index, y=df4h["EMA_slow"], line=dict(color="purple", width=1), name=f"EMA {ema_slow}"))
    fig4h.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig4h, use_container_width=True)

st.subheader("15m Entry Signals (confirmed by 1H & 4H)")
if signals.empty:
    st.info("No recent entry signals found (15m EMA cross with 1H & 4H trend). You can relax filters or change EMA periods / RSI max.")
else:
    st.dataframe(signals.sort_index(ascending=False).head(50))

with st.expander("Show raw 15m data (last 500 rows)"):
    st.dataframe(df15.tail(500))

st.markdown("Notes: Data come from Yahoo Finance intraday 15m. Availability and history length depend on Yahoo. For continuous live trading or longer history use a dedicated market-data API (I ca[...]")
