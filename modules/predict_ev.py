import pickle
import pandas as pd
import numpy as np

def load_ev_model(model_path: str = "../models/model_xgb_ev.pkl") -> object:

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_ev_returns(model: object, news_features: dict, tickers: list) -> pd.DataFrame:
    """
    Predict returns for a list of semiconductor tickers based on extracted news features.

    Args:
        model (object): The pre-trained XGBoost model pipeline.
        news_features (dict): Features extracted from news (sentiment, impact_num, etc.).
        tickers (list): List of ticker symbols to predict.

    Returns:
        pd.DataFrame: DataFrame with columns ['ticker', 'predicted_return'].
    """
    records = []
    for ticker in tickers:
        record = {
            "sentiment":      news_features.get("sentiment", 0.5),
            "impact_num":     news_features.get("impact_num", 0),
            "text_length":    news_features.get("text_length", np.nan),
            "avg_tfidf":      news_features.get("avg_tfidf", np.nan),
            "embedding_mean": news_features.get("embedding_mean", np.nan),
            "ticker":         ticker
        }
        records.append(record)

    df_input = pd.DataFrame(records)
    df_input["predicted_return"] = model.predict(df_input)
    return df_input[["ticker", "predicted_return"]]
