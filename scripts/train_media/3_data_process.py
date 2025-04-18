import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pickle

df = pd.read_csv("media_news_enriched.csv")
df.dropna(subset=["summary"], inplace=True)
df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')

# Generate the impact label column (positive/neutral/negative)
def map_sentiment(score):
    if pd.isna(score):
        return "neutral"
    elif score >= 0.65:
        return "positive"
    elif score <= 0.35:
        return "negative"
    else:
        return "neutral"

df['impact'] = df['sentiment'].apply(map_sentiment)

# Map impact to numerical values: positive=1, neutral=0, negative=-1
impact_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['impact_num'] = df['impact'].map(impact_map)

# Calculate text length for each article
df['text_length'] = df['content'].apply(lambda x: len(str(x)))

# compute the average TF-IDF value per article
vectorizer = TfidfVectorizer(max_features=50)
tfidf_matrix = vectorizer.fit_transform(df['content'])
df['avg_tfidf'] = tfidf_matrix.mean(axis=1).A1  

with open("../../models/tfidf_vectorizer_media.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    
# Compute the average embedding for each article
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = df['content'].apply(lambda x: model_embed.encode(str(x)))
df['embedding_mean'] = df['embedding'].apply(np.mean)

df.to_csv("media_news_clean.csv", index=False)

