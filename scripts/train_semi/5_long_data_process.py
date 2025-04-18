import pandas as pd
import numpy as np
import datetime
import os

def calc_return(df_price, event_date_str,horizon=1):
     # 1. Parse the input date
    try:
        event_date = datetime.datetime.strptime(event_date_str, '%Y-%m-%d').date()
    except Exception:
        return np.nan

    # 2. Ensure df_price.index is a DatetimeIndex of dates in "%Y-%m-%d" format
    #    (you can do this once when loading df_price)
    # df_price.index = pd.to_datetime(df_price['Date']).date

    # 3. Find the first trading day on or after event_date (scan up to 3 days)
    for i in range(4):
        candidate = event_date + datetime.timedelta(days=i)
        cand_str = candidate.strftime('%Y-%m-%d')
        if cand_str in df_price.index:
            row = df_price.loc[cand_str]
            # 4. Compute intraday return
            open_p  = row.get('Open', np.nan)
            close_p = row.get('Close', np.nan)
            if pd.notna(open_p) and open_p != 0:
                return (close_p - open_p) / open_p
            else:
                return np.nan

    # If no trading day found in the next 3 calendar days
    return np.nan


def build_long_format_dataset_new(news_csv: str, stock_csv_list: list, horizon: int = 5) -> pd.DataFrame:
    df_news = pd.read_csv(news_csv)
    
    # Convert publishedAt to a proper date format ("YYYY-MM-DD")
    df_news['publish_date'] = pd.to_datetime(
        df_news['publishedAt'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce'
    ).dt.strftime('%Y-%m-%d')
    df_news = df_news.dropna(subset=['publish_date'])
    
    # For news, we assume the following columns exist; if not, they will be set as NaN.
    required_news_cols = ['sentiment', 'impact', 'impact_num', 'text_length', 'avg_tfidf', 'embedding_mean']
    for col in required_news_cols:
        if col not in df_news.columns:
            df_news[col] = np.nan
            
    # Compute interaction feature if needed 
    df_news['sentiment_impact'] = df_news['sentiment'] * df_news['impact_num']
    
    long_data = []
    # Process each news article individually 
    for idx, news_row in df_news.iterrows():
        news_date = news_row['publish_date']
        sentiment_val = news_row['sentiment']
        impact_val = news_row['impact']
        impact_num_val = news_row['impact_num']
        text_length_val = news_row['text_length']
        avg_tfidf_val = news_row['avg_tfidf']
        embedding_mean_val = news_row['embedding_mean']
        
        
        # Process each stock file
        for stock_csv in stock_csv_list:
            df_stock = pd.read_csv(stock_csv)
            df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
            df_stock = df_stock.dropna(subset=['Date'])
            # Set index as "YYYY-MM-DD"
            df_stock.set_index(df_stock['Date'].dt.strftime('%Y-%m-%d'), inplace=True)
            
            for col in ['momentum', 'daily_return', 'volatility']:
                df_stock[col] = pd.to_numeric(df_stock[col], errors='coerce')
                df_stock[col].fillna(df_stock[col].mean(), inplace=True)
                
            # Extract ticker from file name
            ticker = os.path.basename(stock_csv).split('_')[0]
            
            # Calculate target_return using calc_return for the news_date.
            ret_val = calc_return(df_stock, news_date, horizon=horizon)
            if np.isnan(ret_val):
                continue  
            
            # Extract stock's technical indicators on the news_date: momentum, daily_return, volatility.
            if news_date in df_stock.index:
                stock_row = df_stock.loc[news_date]
                momentum_val = stock_row.get('momentum', np.nan)
                daily_return_val = stock_row.get('daily_return', np.nan)
                volatility_val = stock_row.get('volatility', np.nan)
            else:
                continue
            
            # Prepare a dictionary with exactly the required columns.
            record = {
                'date': news_date,
                'ticker': ticker,
                'sentiment': sentiment_val,
                'impact': impact_val,
                'impact_num': impact_num_val,
                'text_length': text_length_val,
                'avg_tfidf': avg_tfidf_val,
                'embedding_mean': embedding_mean_val,
                'momentum': momentum_val,
                'daily_return': daily_return_val,
                'volatility': volatility_val,
                'target_return': ret_val
            }
            long_data.append(record)
    
    df_long = pd.DataFrame(long_data)
    # Ensure the DataFrame contains the exact order of columns as required.
    output_columns = ['date', 'ticker', 'sentiment', 'impact', 'impact_num',
                      'text_length', 'avg_tfidf', 'embedding_mean',
                      'momentum', 'daily_return', 'volatility', 'target_return']
    df_long = df_long[output_columns]
    
    return df_long

if __name__ == '__main__':
    # Define paths for news CSV and stock CSVs (adjust according to your file structure).
    news_csv_path = "semi_news_clean.csv"  
    stock_csv_list = [
        "../../data/semi/NVDA_1mo.csv",
        "../../data/semi/AMD_1mo.csv",
        "../../data/semi/AMAT_1mo.csv",
        "../../data/semi/AVGO_1mo.csv",
        "../../data/semi/INTC_1mo.csv",
        "../../data/semi/KLAC_1mo.csv",
        "../../data/semi/LRCX_1mo.csv",
        "../../data/semi/MCHP_1mo.csv",
        "../../data/semi/MU_1mo.csv",
        "../../data/semi/NXPI_1mo.csv",
        "../../data/semi/QCOM_1mo.csv",
        "../../data/semi/SWKS_1mo.csv",
        "../../data/semi/ADI_1mo.csv",
        "../../data/semi/ON_1mo.csv",
        "../../data/semi/TER_1mo.csv",
        "../../data/semi/TXN_1mo.csv",
        "../../data/semi/TSM_1mo.csv",
        "../../data/semi/UMC_1mo.csv",
        "../../data/semi/ASML_1mo.csv",
        "../../data/semi/STM_1mo.csv",
        
    ]
    
    horizon_days = 1
    
    print("Building long-format dataset (single news per sample with specified columns)...")
    df_long = build_long_format_dataset_new(news_csv_path, stock_csv_list, horizon=horizon_days)
    print(f"Dataset built, number of samples: {df_long.shape[0]}")
    print(df_long.head())
    
   
    df_long.to_csv("long_data.csv", index=False, encoding='utf-8-sig')

