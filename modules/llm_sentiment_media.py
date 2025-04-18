import json
import os
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

client = OpenAI(
    api_key      = st.secrets["openai"]["api_key"],
    organization = st.secrets["openai"]["organization"],
    project      = st.secrets["openai"]["project"],
)


def analyze_news_with_llm(news_text: str) -> dict:

    prompt = f"""
You are an AI assistant that analyzes news for its relevance to the Digital Entertainment & Media sector,
including streaming video platforms, online gaming, esports, digital music, podcasts, and emerging
interactive media experiences (e.g., virtual concerts, VR entertainment).

Step 1: Decide whether the article is relevant to Digital Entertainment & Media.
  - Relevant includes:
      * Streaming (OTT, SVOD, AVOD), on‑demand video, live streaming.
      * Online gaming, esports tournaments, game publishing deals.
      * Digital music streaming, podcast platforms, virtual concerts.
      * Regulations or policies that affect digital content distribution, royalties, or advertising revenue.
  - If NOT relevant, output exactly: {{"sentiment":0.5,"impact_num":0}} and STOP.

Step 2: If relevant, compute:
  1. "sentiment": float in [0.0, 1.0] where 0.0 = very negative and 1.0 = very positive.
  2. "impact_num": integer
       -  1 if sentiment >= 0.65
       - -1 if sentiment <= 0.35
       -  0 otherwise

ONLY output a single line of valid JSON with the two keys "sentiment" and "impact_num".

News: {news_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=50,
            stream=False
        )
        # Extract the response content using attribute access.
        response_text = response.choices[0].message.content
        response_text = response_text.replace("'", "\"")
        result = json.loads(response_text)
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        # Return default values if any error occurs.
        result = {"sentiment": 0.5, "impact_num": 0}
    return result

def compute_avg_tfidf(news_text: str) -> float:
    """
    Compute the average TF-IDF value for the given news text using a TfidfVectorizer with max_features=50.
    This method exactly matches the one used during training.
    
    Parameters:
        news_text (str): The news text.
        
    Returns:
        float: The average TF-IDF value.
    """
    import pickle
    with open("models/tfidf_vectorizer_media.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = vectorizer.transform([news_text])
    avg_tfidf = tfidf_matrix.mean(axis=1).A1[0]
    return float(avg_tfidf)

model_embed = SentenceTransformer('all-MiniLM-L6-v2')
def compute_embedding_mean(news_text: str) -> float:
    """
    Compute the mean of the sentence embedding for the given news text using SentenceTransformer 'all-MiniLM-L6-v2'.
    This method matches exactly the embedding calculation used during training.
    
    Parameters:
        news_text (str): The news text.
        
    Returns:
        float: The mean value of the sentence embedding.
    """
    embedding = model_embed.encode(news_text)
    return float(np.mean(embedding))

def extract_additional_features(news_text: str) -> dict:
    """
    Extract additional features from the news text:
      - text_length: The length (number of characters) of the news text.
      - avg_tfidf: The average TF-IDF score computed using a TfidfVectorizer.
      - embedding_mean: The mean value of the sentence embedding.
    
    Parameters:
        news_text (str): The news text.
        
    Returns:
        dict: A dictionary with keys "text_length", "avg_tfidf", and "embedding_mean".
    """
    features = {}
    features['text_length'] = len(news_text)
    features['avg_tfidf'] = compute_avg_tfidf(news_text)
    features['embedding_mean'] = compute_embedding_mean(news_text)
    return features

def analyze_and_extract_features(news_text: str) -> dict:
    """
    Perform LLM analysis on the news text and extract additional features.
    
    Parameters:
        news_text (str): The news text.
        
    Returns:
        dict: A dictionary combining LLM analysis results and extracted features.
              The keys include "sentiment", "impact_num", "text_length", "avg_tfidf", and "embedding_mean".
    """
    llm_result = analyze_news_with_llm(news_text)
    additional_features = extract_additional_features(news_text)
    combined_features = {**llm_result, **additional_features}
    return combined_features

def llm_event_analysis(news_text: str, pred_lines: str) -> str:
    """
    Given raw news_text and a preformatted pred_lines string (each line "- TICKER: +X.XX%"),
    call the LLM to generate a detailed (~200 words) sector impact analysis.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a senior financial analyst specializing in digital media. "
            "Given a news article and a list of predicted intraday returns for individual digital media stocks, "
            "produce a comprehensive impact analysis of approximately 200 words. "
            "Discuss the overall effect on the digital media sector, including streaming trends, advertising dynamics, content monetization drivers, "
            "and highlight any stock with an unusually high or low predicted return—explaining the likely reason for its outperformance or underperformance."
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"News article:\n{news_text}\n\n"
            "Predicted intraday returns:\n"
            f"{pred_lines}\n\n"
            "Please provide your detailed sector impact analysis."
        )
    }

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[system_msg, user_msg],
        temperature=0.7,
        max_tokens=400  
    )
    return resp.choices[0].message.content.strip()
