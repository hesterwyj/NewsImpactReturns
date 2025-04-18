import yfinance as yf
import pandas as pd
import numpy as np
import datetime

def fetch_stock_data(ticker, period='1mo'):
    """
    Fetch stock data for the given ticker using yfinance.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        period (str): Period for which data is fetched (default '1mo').
    
    Returns:
        DataFrame: Stock data with columns such as Open, High, Low, Close, Volume, etc.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def calculate_momentum(df, window=5):
    """
    Calculate the momentum as the percentage change in the 'Close' price over a specified window.
    
    Parameters:
        df (DataFrame): Stock price DataFrame.
        window (int): Number of periods over which to calculate momentum (default is 5).
    
    Returns:
        DataFrame: The input DataFrame with a new column 'momentum' added.
    """
    df['momentum'] = df['Close'].pct_change(periods=window)
    return df

def calculate_volatility(df, window=5):
    """
    Calculate the rolling volatility (standard deviation) of daily returns over a specified window.
    
    Parameters:
        df (DataFrame): Stock price DataFrame.
        window (int): Rolling window size for volatility calculation (default is 5).
    
    Returns:
        DataFrame: The input DataFrame with a new column 'volatility' added.
    """
    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=window).std()
    return df

if __name__ == '__main__':
    # List of semiconductor stock tickers (modify according to your research)
    tickers = [
       "AMZN",  # Amazon
        "BABA",  # Alibaba Group
        "PDD",   # PDD Holdings
        "JD",    # JD.com
        "MELI",  # MercadoLibre
        "SHOP",  # Shopify
        "SE",    # Sea Limited
        "EBAY",  # eBay
        "ETSY",  # Etsy
        "DASH",  # DoorDash
        "UBER",  # Uber (delivery & ride-hailing platform)
        "BKNG",  # Booking Holdings (online travel)
        "TCOM",  # Trip.com
        "W",     # Wayfair
        "AFRM",   # Affirm
        "CHWY",    # Chewy (pet e‑commerce)
        "PYPL",  # PayPal
        "NTES",  # NetEase (games + e‑commerce subsidiary)
        "GLBE",  # Global‑e Online
        "FVRR"   # Fiverr (digital marketplace)
    ]
    
    # Loop over each ticker, fetch data, calculate momentum and volatility, and save to CSV.
    for ticker in tickers:
        df = fetch_stock_data(ticker, period='1mo')
        if df.empty:
            print(f"No data for {ticker}.")
            continue
        
        # Calculate momentum (percentage change over the past 5 days)
        df = calculate_momentum(df, window=5)
        # Calculate volatility (rolling 5-day standard deviation of daily returns)
        df = calculate_volatility(df, window=5)
        
        output_file = f'../../data/int/{ticker}_1mo.csv'
        df.to_csv(output_file)
        
