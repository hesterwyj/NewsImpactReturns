import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import streamlit as st 

API_KEY = st.secrets["newsapi"]["api_key"]
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

query = (
    "semiconductor industry OR chip industry OR semiconductor OR "
    "global semiconductor OR US chip industry OR European semiconductor OR "
    "integrated circuit OR fabless OR foundry OR chip manufacturing OR IC design"
)

# Set time range: fetch news from the past 30 days
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

# Maximum number of news articles per day
MAX_RESULTS_PER_DAY = 100

selected_articles = []

current_date = start_date
while current_date <= end_date:
    day_str = current_date.strftime('%Y-%m-%d')
    params = {
        "q": query,
        "apiKey": API_KEY,
        "from": day_str,
        "to": day_str,
        "language": "en",
        "pageSize": MAX_RESULTS_PER_DAY,
        "sortBy": "publishedAt"
    }
    
    response = requests.get(NEWSAPI_ENDPOINT, params=params)
    if response.status_code != 200:
        print(f"❌ Failed to fetch data on {day_str}, status code: {response.status_code}")
        current_date += timedelta(days=1)
        continue
        
    data = response.json()
    articles = data.get("articles", [])
    if not articles:
        print(f"⚠️ No articles found on {day_str}")
        current_date += timedelta(days=1)
        continue

    for article in articles:
        article["fetch_date"] = day_str
        selected_articles.append(article)
    
    print(f"✅ Fetched {len(articles)} articles on {day_str}")
    time.sleep(1)  
    current_date += timedelta(days=1)

df = pd.DataFrame(selected_articles)
csv_filename = "newsapi_semi_news.csv"
df.to_csv(csv_filename, index=False)
