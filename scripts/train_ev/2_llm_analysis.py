import pandas as pd
from openai import OpenAI
import streamlit as st
import streamlit as st
client = OpenAI(
    api_key      = st.secrets["openai"]["api_key"],
    organization = st.secrets["openai"]["organization"],
    project      = st.secrets["openai"]["project"],
)


def analyze_news_with_llm(news_text):
    """
    Call GPT-4o-mini to process the news text:
    - Generate a brief summary (1–2 sentences)
    - Return a sentiment score between 0 and 1 (0 = negative, 1 = positive)
    """
    user_prompt = f"""
    You are an AI assistant that analyzes news for its relevance to the New Energy Vehicle (NEV) industry,
    including direct mentions of vehicle models, OEMs, EV components, battery technology,
    charging infrastructure, as well as macro-level policies (e.g., subsidies, emissions regulations,
    tariffs on EV imports, supply‑chain disruptions).

    Step 1: Decide if the article is relevant to the NEV industry.
    - Relevant includes:
        * Mentions of electric vehicles or manufacturers (e.g., Tesla, NIO, BYD).
        * Discussions of battery technology, charging infrastructure, range, BMS.
        * Policy actions (subsidies, tax credits, import/export tariffs) affecting EV adoption.
    - If NOT relevant, output exactly: {{"sentiment":0.5,"impact_num":0}} and STOP.
    
    Step 2: If relevant, compute:
    1. "sentiment": a float between 0.0 (very negative) and 1.0 (very positive) reflecting the tone.
    2. "summary": a concise 1-2 sentence summary of the main point.

    DO NOT include any explanations or extra text.
    Just return a single line of valid JSON.

    News:
    {news_text}
    """


    # Use streaming output
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_prompt}],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            full_response += delta_content

    return full_response

if __name__ == '__main__':
    df = pd.read_csv("newsapi_ev_news.csv")

    summaries = []
    sentiments = []

    # Loop through the news records and call the LLM
    for idx, row in df.iterrows():
        content = str(row.get("content",""))
        if not content.strip():
            # If content is empty, skip or use default values
            summaries.append("")
            sentiments.append("")
            continue

        response_text = analyze_news_with_llm(content[:4000])
        
        import json
        try:
            result = json.loads(response_text)
            summary = result.get("summary", "")
            sentiment = result.get("sentiment", "")
        except:
            summary = "Failed to parse"
            sentiment = ""

        summaries.append(summary)
        sentiments.append(sentiment)

        print(f"[{idx}] Done. Summary: {summary[:60]}...")

    
    df["summary"] = summaries
    df["sentiment"] = sentiments
    df.to_csv("ev_news_enriched.csv", index=False)

