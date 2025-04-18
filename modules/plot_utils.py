import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

@st.cache_data(show_spinner=False)
def get_history_returns(ticker: str, days: int = 14):
    """
    Download historical data for the most recent business days and compute daily returns.
    Adjust 'today' to the last business day if it's a weekend.
    Returns a tuple:
      - pd.Series of daily_return indexed by normalized dates.
      - pd.Timestamp of the reference date (the last trading day considered).
    """
    # Determine reference date: if today is weekend, roll back to last business day
    today = pd.Timestamp.now().normalize()
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        ref_date = today - BDay(1)
    else:
        ref_date = today
    # Calculate start and end dates for download
    start_date = ref_date - BDay(days)
    end_date = ref_date + BDay(1)

    # Download from Yahoo Finance
    hist = yf.download(
        ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )
    # Ensure valid data
    if "Close" not in hist.columns or len(hist) < 2:
        return pd.Series(dtype=float), ref_date

    # Compute daily returns
    hist = hist.copy()
    hist["daily_return"] = hist["Close"].pct_change()
    returns = hist["daily_return"].dropna()
    # Normalize timestamps to date only
    returns.index = returns.index.normalize()
    return returns, ref_date


def plot_returns_with_prediction(returns: pd.Series, pred_val: float, ticker: str, ref_date: pd.Timestamp):
    """
    Build a Plotly figure combining historical returns and a future prediction:
      - Black solid line + markers for historical returns (last 7 business days)
      - Colored dashed line + diamond marker for prediction on ref_date+1 business day
      Format hover and axes as percentages.

    Parameters:
      returns: pd.Series of daily returns indexed by date.
      pred_val: predicted return for next business day.
      ticker: stock ticker.
      ref_date: last trading day used as reference.
    """
    # Historical returns: last 7 business days
    hist7 = returns.tail(7)
    if hist7.empty:
        return None
    # Remove duplicate dates if any
    hist7 = hist7[~hist7.index.duplicated(keep='first')]

    # Determine prediction date: next business day after ref_date
    next_date = ref_date + BDay(1)

    # Prepare plot arrays
    actual_x = hist7.index
    actual_y = hist7.values
    pred_x = [hist7.index[-1], next_date]
    pred_y = [actual_y[-1], pred_val]
    pred_color = 'green' if pred_val >= 0 else 'red'

    # Build figure
    fig = go.Figure()
    # Actual segment
    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_y,
        mode='lines+markers', name='Actual',
        line=dict(color='black', dash='solid'),
        marker=dict(color='black', size=6),
        hovertemplate='%{y:.2%}<extra></extra>'
    ))
    # Predicted segment
    fig.add_trace(go.Scatter(
        x=pred_x, y=pred_y,
        mode='lines+markers', name='Predicted',
        line=dict(color=pred_color, dash='dash'),
        marker=dict(color=pred_color, size=8, symbol='diamond'),
        hovertemplate='%{y:.2%}<extra></extra>'
    ))

    # Layout and formatting
    fig.update_layout(
        title=f"{ticker} Daily Returns + Prediction",
        xaxis_title="Date", yaxis_title="Return",
        legend=dict(y=0.99, x=0.01)
    )
    fig.update_xaxes(tickformat="%Y-%m-%d")
    fig.update_yaxes(tickformat=".2%")
    return fig
