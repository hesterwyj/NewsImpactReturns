#sector_utils.py

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def fetch_performance(etf: str, index: str, period: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for label, ticker in {etf: etf, index: index}.items():
        hist = yf.Ticker(ticker).history(period=period)["Close"]
        df[label] = (hist / hist.iloc[0] - 1) * 100
    return df

@st.cache_data
def fetch_returns(etf: str, index: str) -> dict:
    t_etf, t_idx = yf.Ticker(etf), yf.Ticker(index)
    prc_etf = t_etf.history(period="1d")["Close"]
    prc_idx = t_idx.history(period="1d")["Close"]
    prev_etf = t_etf.info.get("previousClose", np.nan)
    prev_idx = t_idx.info.get("previousClose", np.nan)
    # day returns
    day_etf = (prc_etf.iloc[-1]/prev_etf - 1) if prev_etf else np.nan
    day_idx = (prc_idx.iloc[-1]/prev_idx - 1) if prev_idx else np.nan

    def hor(tkr, per):
        h = yf.Ticker(tkr).history(period=per)["Close"]
        return (h.iloc[-1]/h.iloc[0] - 1) if len(h)>=2 else np.nan

    keys = ["ytd","1y","3y","5y"]
    out = {"day": day_etf, "day_idx": day_idx}
    for k in keys:
        out[k]       = hor(etf, k)
        out[f"{k}_idx"] = hor(index, k)
    return out

def render_sector_header(title: str, description: str):
    """Render the header and description."""
    st.header(title)
    st.write(description)
    st.markdown("---")

def render_performance_chart(title: str, etf: str, index: str="^GSPC"):
    """Render the performance vs S&P 500 line chart."""
    st.subheader(f"{etf} vs S&P 500 Performance")
    period = st.select_slider(
        f"Select period for {title}",
        options=["1d","5d","1mo","6mo","ytd","1y","5y","max"],
        value="5y",
        key=f"{title}_period"
    )
    df = fetch_performance(etf, index, period)
    st.line_chart(df, height=350, use_container_width=True)

def render_return_cards(etf: str, index: str="^GSPC"):
    """Render the five ‘Returns Summary’ cards."""
    st.subheader("Returns Summary")
    rets = fetch_returns(etf, index)
    horizons = [
        ("Day Return", "day"),
        ("YTD Return", "ytd"),
        ("1-Year Return", "1y"),
        ("3-Year Return", "3y"),
        ("5-Year Return", "5y"),
    ]
    cols = st.columns(len(horizons))
    for (label,key), col in zip(horizons, cols):
        ind = rets[key]; idx = rets[f"{key}_idx"]
        col.markdown(f"**{label}**")
        col.markdown(
            f"{etf}  <span style='color:{'green' if ind>=0 else 'red'}'>{ind*100:+.2f}%</span>",
            unsafe_allow_html=True
        )
        col.markdown(
            f"S&P 500  <span style='color:{'green' if idx>=0 else 'red'}'>{idx*100:+.2f}%</span>",
            unsafe_allow_html=True
        )
    st.markdown("---")
