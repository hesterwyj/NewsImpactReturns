import streamlit as st
import pandas as pd
import json
import os
import pickle
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
import plotly.express as px
from modules.plot_utils import get_history_returns, plot_returns_with_prediction
from modules.sector_utils import (
    render_sector_header,
    render_performance_chart,
    render_return_cards,
)


# 侧边栏导航
# Sidebar navigation using radio buttons
st.sidebar.title("Tech Dashboard")
page = st.sidebar.radio("Select a Page", [
    "Overview", "Semiconductor", "EV & Battery",
    "Internet & E‑commerce", "Digital Media"
])  # st.radio :contentReference[oaicite:4]{index=4}


# 第一页
if page == "Overview":
    st.title("Technology Stocks Overview")

    # 1. Metric row for current price & daily % change
    tickers = {
        "Semiconductor (SOXX)": "SOXX",
        "EV & Battery (LIT)":   "LIT",
        "Internet & E‑comm (FDN)": "FDN",
        "Digital Media (PBS)":  "PBS",
        "S&P 500 (^GSPC)":      "^GSPC"
    }
    cols = st.columns(len(tickers))
    for col, (name, tk) in zip(cols, tickers.items()):
        ticker = yf.Ticker(tk)
        info = ticker.info
        prev = info.get("previousClose", None)
        hist = ticker.history(period="1d")["Close"]
        if prev and len(hist) > 0:
            price = hist.iloc[-1]
            pct   = (price/prev - 1) * 100
            col.metric(name, f"{price:.2f}", f"{pct:+.2f}%")
        else:
            col.metric(name, "–", "–")

    st.markdown("---")

    # 2. Performance chart over selectable period
    st.subheader("Sector ETF Performance Comparison")
    period = st.select_slider(
        "Select period",
        options=["1mo", "6mo", "ytd", "1y"],
        value="6mo",
        key="overview_perf_period"
    )
    perf = pd.DataFrame()
    for name, tk in tickers.items():
        series = yf.Ticker(tk).history(period=period)["Close"]
        perf[name] = (series / series.iloc[0] - 1) * 100
    perf.index = pd.to_datetime(perf.index)
    st.line_chart(perf, use_container_width=True, height=400)

    st.markdown("---")

    # 3. Data table: Market Cap & YTD Return

    st.subheader("ETF Market Cap & YTD Return")
    rows = []
    for name, tk in tickers.items():
        info = yf.Ticker(tk).info
        ytd_pct = perf[name].iloc[-1]  
        rows.append({
            "ETF":             name,
            "Market Cap (USD)": info.get("totalAssets") or info.get("marketCap"),
            "YTD Return (%)":   ytd_pct
        })

    df_table = pd.DataFrame(rows)
    df_table["Market Cap (USD)"] = df_table["Market Cap (USD)"].map(
        lambda x: f"{x/1e9:.1f} B" if x else "–"
    )
    df_table["YTD Return (%)"] = df_table["YTD Return (%)"].map("{:+.2f}%".format)

    st.dataframe(df_table.set_index("ETF"), use_container_width=True)


   
# 半导体
elif page == "Semiconductor":
    from modules.llm_sentiment_semi import llm_event_analysis, analyze_and_extract_features
    title = "Semiconductors"
    description = (
            "Semiconductor companies that design, manufacture, and market integrated circuits, "
            "microprocessors, logic devices, chipsets, and memory chips for a wide variety of users."
        )
    etf_ticker = "SOXX"
    render_sector_header(title, description)
    render_performance_chart(title, etf_ticker)
    render_return_cards(etf_ticker)

    # Imports for this page
    from modules.predict_semi import load_semi_model, predict_semi_returns

    # Cache and load the model once
    @st.cache_resource
    def get_model():
        return load_semi_model("models/model_xgb_semi.pkl")
    model_semi = get_model()

    # A. News input
    news_text = st.text_area("Enter semiconductor news article:", height=150)

    # B. Analyze & Predict button
    if st.button("Analyze & Predict"):
        # 1) LLM sentiment + text features
        features = analyze_and_extract_features(news_text)
        st.subheader("Extracted Features")
        st.write(features)

        # 2) Predict returns for all tickers
        tickers = [
            "NVDA","AMD","INTC","TSM","ASML","QCOM",
            "AMAT","LRCX","KLAC","MU","NXPI","AVGO",
            "MCHP","TER","SWKS","ON","ADI","STM","UMC"
        ]
        df_pred = predict_semi_returns(model_semi, features, tickers)
        
        # 3) Store predictions in session state
        st.session_state["semi_df_pred"] = df_pred
        st.session_state["semi_news"] = news_text
    
        # 4) Download button
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full results as CSV", 
            csv, 
            "semi_predictions.csv", 
            "text/csv"
        )

    # C. Plot returns + prediction if predictions exist
    if "semi_df_pred" in st.session_state:
        df_pred = st.session_state["semi_df_pred"]
        st.subheader("Predicted Returns")
        st.table(df_pred[["ticker", "predicted_return"]])
        
        st.subheader("Return Timeline for Selected Stock")

        # 1) Stock selector with unique key
        selected = st.selectbox(
            "Choose a stock to visualize:", 
            df_pred["ticker"].tolist(), 
            key="semi_plot_selector"
        )

        # 2) Fetch past returns (last 14 calendar days → daily returns)
        returns, ref_date = get_history_returns(selected, days=14)

        # 3) Extract today's predicted return
        pred_val = float(
            df_pred.loc[df_pred["ticker"] == selected, "predicted_return"])

        # 4) Build and render the combined line chart
        fig = plot_returns_with_prediction(returns, pred_val, selected, ref_date)
        st.plotly_chart(fig, use_container_width=True)
    
    # D. Event Commentary
    st.subheader("Event Commentary")

    if "semi_df_pred" in st.session_state:
        news = st.session_state["semi_news"]
        df_pred = st.session_state["semi_df_pred"]
   # 1) Sector‐level metrics
        avg_ret = df_pred["predicted_return"].mean() * 100
        pos_cnt = (df_pred["predicted_return"] > 0).sum()
        neg_cnt = (df_pred["predicted_return"] < 0).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sector Avg Return", f"{avg_ret:+.2f}%")
        c2.metric("Stocks ↑", pos_cnt)
        c3.metric("Stocks ↓", neg_cnt)
        st.markdown("---")
    # 3) Generate detailed commentary
    if "semi_df_pred" in st.session_state:
        news = st.session_state["semi_news"]
        df_pred = st.session_state["semi_df_pred"]

        if st.button("Generate Detailed Commentary"):
            # Format predictions for the LLM prompt
            pred_lines = "\n".join(
                f"- {row.ticker}: {row.predicted_return * 100:+.2f}%"
                for row in df_pred.itertuples()
            )

            # Call the LLM for a ~200‑word analysis
            with st.spinner("Generating detailed sector impact analysis…"):
                commentary = llm_event_analysis(news, pred_lines)

            # Display the detailed commentary
            st.subheader("Detailed Sector Impact Commentary")
            st.write(commentary)

# 新能源
# EV & Battery
elif page == "EV & Battery":
    # Import LLM analysis functions for EV & Battery
    from modules.llm_sentiment_ev import llm_event_analysis as llm_ev_analysis, analyze_and_extract_features as analyze_ev_features

    # Section header
    title = "EV & Battery"
    description = (
        "Electric vehicle and battery companies, including automakers, "
        "battery material suppliers, and charging infrastructure providers."
    )
    etf_ticker = "LIT"  # Global X Lithium & Battery Tech ETF
    render_sector_header(title, description)
    render_performance_chart(title, etf_ticker)
    render_return_cards(etf_ticker)

    # Import prediction tools for EV & Battery
    from modules.predict_ev import load_ev_model, predict_ev_returns

    # Cache and load the EV model
    @st.cache_resource
    def get_ev_model():
        return load_ev_model("models/model_xgb_ev.pkl")
    model_ev = get_ev_model()

    # A. News input
    ev_news = st.text_area(
    "Enter EV & Battery news article:",
    height=150,
    key="ev_news")

    # B. Analyze & Predict button
    if st.button("Analyze & Predict EV"):
        # 1) LLM sentiment + text features
        ev_features = analyze_ev_features(ev_news)
        st.subheader("Extracted Features")
        st.write(ev_features)

        # 2) Predict returns for all EV tickers
        ev_tickers = [
                "TSLA", "NIO", "LI", "XPEV", "LCID", "RIVN", "BYDDY", "BYDDF",
                "WKHS", "ALB", "LAC", "SQM", "MP", "PLUG", "BLDP", "FCEL",
                 "QS", "APTV", "CHPT"
        ]
        ev_df_pred = predict_ev_returns(model_ev, ev_features, ev_tickers)

        # 3) Store in session state
        st.session_state["ev_df_pred"] = ev_df_pred
        # st.session_state["ev_news"]    = ev_news

        # 4) Download button
        csv_ev = ev_df_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download EV predictions CSV",
            csv_ev,
            "ev_predictions.csv",
            "text/csv"
        )

    # C. Plot returns + prediction if available
    if "ev_df_pred" in st.session_state:
        ev_df_pred = st.session_state["ev_df_pred"]
        st.subheader("Predicted EV & Battery Returns")
        st.table(ev_df_pred[["ticker", "predicted_return"]])

        st.subheader("Return Timeline for Selected EV Stock")
        selected_ev = st.selectbox(
            "Choose a stock to visualize:",
            ev_df_pred["ticker"].tolist(),
            key="ev_plot_selector"
        )
        returns_ev, ref_date_ev = get_history_returns(selected_ev, days=14)
        pred_val_ev = float(ev_df_pred.loc[ev_df_pred["ticker"] == selected_ev, "predicted_return"])
        fig_ev = plot_returns_with_prediction(returns_ev, pred_val_ev, selected_ev, ref_date_ev)
        st.plotly_chart(fig_ev, use_container_width=True)

    # D. Event Commentary
    st.subheader("Event Commentary")

    # Only proceed if EV predictions exist
    if "ev_df_pred" in st.session_state:
        # Safely read EV news & predictions
        ev_news    = st.session_state.get("ev_news", "")
        ev_df_pred = st.session_state["ev_df_pred"]

        # 1) Sector-level metrics for EV & Battery
        avg_ret = ev_df_pred["predicted_return"].mean() * 100
        pos_cnt = (ev_df_pred["predicted_return"] > 0).sum()
        neg_cnt = (ev_df_pred["predicted_return"] < 0).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sector Avg Return", f"{avg_ret:+.2f}%")
        c2.metric("Stocks ↑", pos_cnt)
        c3.metric("Stocks ↓", neg_cnt)
        st.markdown("---")

        # 2) Generate detailed EV commentary
        if ev_news and st.button("Generate EV Commentary", key="btn_comment_ev"):
            # a) Build prediction lines
            ev_pred_lines = "\n".join(
                f"- {row.ticker}: {row.predicted_return * 100:+.2f}%"
                for row in ev_df_pred.itertuples()
            )
            # b) Call the LLM
            with st.spinner("Generating detailed EV & Battery impact analysis…"):
                ev_commentary = llm_ev_analysis(ev_news, ev_pred_lines)
            # c) Display result
            st.subheader("Detailed EV & Battery Impact Commentary")
            st.write(ev_commentary)
    else:
        st.info("Please click 'Analyze & Predict EV' above to generate predictions first.")

# 互联网与电子商务
elif page == "Internet & E‑commerce":
    # Import LLM analysis functions for IE
    from modules.llm_sentiment_int import (
        llm_event_analysis as llm_int_analysis,
        analyze_and_extract_features as analyze_int_features
    )

    # Section header
    title = "Internet & E‑commerce"
    description = (
        "Internet platforms, online retailers, and e‑commerce services, "
        "including marketplaces, digital advertising, and cloud‑based commerce tools."
    )
    etf_ticker = "FDN"
    render_sector_header(title, description)
    render_performance_chart(title, etf_ticker)
    render_return_cards(etf_ticker)

    # Import prediction tools for IE
    from modules.predict_int import load_int_model, predict_int_returns

    # Cache and load the IE model
    @st.cache_resource
    def get_int_model():
        return load_int_model("models/model_xgb_int.pkl")
    model_int = get_int_model()

    # A. News input
    int_news = st.text_area(
        "Enter Internet & E‑commerce news article:",
        height=150,
        key="int_news"
    )

    # B. Analyze & Predict button
    if st.button("Analyze & Predict IE"):
        # 1) LLM sentiment + text features
        int_features = analyze_int_features(int_news)
        st.subheader("Extracted Features")
        st.write(int_features)

        # 2) Predict returns for all IE tickers
        int_tickers = [
                "AMZN", "BABA", "PDD", "JD", "MELI", "SHOP", "SE", "EBAY",
                "ETSY", "DASH", "UBER", "BKNG", "TCOM", "W", "AFRM",
                 "PYPL", "GLBE", "NTES", "CHWY", "FVRR"
        ]
        int_df_pred = predict_int_returns(model_int, int_features, int_tickers)

        # 3) Store in session state
        st.session_state["int_df_pred"] = int_df_pred
        # (int_news 已由 widget 写入，无需手动赋值)

        # 4) Download button
        csv_int = int_df_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download IE predictions CSV",
            csv_int,
            "ie_predictions.csv",
            "text/csv"
        )

    # C. Plot returns + prediction if available
    if "int_df_pred" in st.session_state:
        int_df_pred = st.session_state["int_df_pred"]
        st.subheader("Predicted Internet & E‑commerce Returns")
        st.table(int_df_pred[["ticker", "predicted_return"]])

        st.subheader("Return Timeline for Selected IE Stock")
        selected_int = st.selectbox(
            "Choose a stock to visualize:",
            int_df_pred["ticker"].tolist(),
            key="int_plot_selector"
        )
        returns_int, ref_date_int = get_history_returns(selected_int, days=14)
        pred_val_int = float(
            int_df_pred.loc[int_df_pred["ticker"] == selected_int, "predicted_return"]
        )
        fig_int = plot_returns_with_prediction(
            returns_int, pred_val_int, selected_int, ref_date_int
        )
        st.plotly_chart(fig_int, use_container_width=True)

    # D. Event Commentary
    st.subheader("Event Commentary")

    if "int_df_pred" in st.session_state:
        int_news = st.session_state.get("int_news", "")
        int_df_pred = st.session_state["int_df_pred"]
        
        # 1) Sector‐level metrics
        avg_ret = int_df_pred["predicted_return"].mean() * 100
        pos_cnt = (int_df_pred["predicted_return"] > 0).sum()
        neg_cnt = (int_df_pred["predicted_return"] < 0).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sector Avg Return", f"{avg_ret:+.2f}%")
        c2.metric("Stocks ↑", pos_cnt)
        c3.metric("Stocks ↓", neg_cnt)
        st.markdown("---")
        # 2) Generate detailed commentary
        if int_news and st.button("Generate IE Commentary"):
            # Format predictions for prompt
            int_pred_lines = "\n".join(
                f"- {row.ticker}: {row.predicted_return * 100:+.2f}%"
                for row in int_df_pred.itertuples()
            )
            # Call the LLM for a ~200‑word analysis
            with st.spinner("Generating detailed Internet & E‑commerce impact analysis…"):
                int_commentary = llm_int_analysis(int_news, int_pred_lines)

            st.subheader("Detailed IE Impact Commentary")
            st.write(int_commentary)
    else:
        st.info("Please click 'Analyze & Predict IE' above to generate predictions first.")

# Digital Media
elif page == "Digital Media":
    # 1. Import LLM analysis functions for Digital Media
    from modules.llm_sentiment_media import (
        llm_event_analysis as llm_media_analysis,
        analyze_and_extract_features as analyze_media_features
    )

    # 2. Section header
    title = "Digital Media"
    description = (
        "Digital media companies including streaming services, "
        "digital advertising platforms, content creators, and social media networks."
    )
    etf_ticker = "PBS"  # Invesco Dynamic Media ETF
    render_sector_header(title, description)
    render_performance_chart(title, etf_ticker)
    render_return_cards(etf_ticker)

    # 3. Import prediction tools for Digital Media
    from modules.predict_media import load_media_model, predict_media_returns

    # Cache and load the Digital Media model
    @st.cache_resource
    def get_media_model():
        return load_media_model("models/model_xgb_media.pkl")
    model_media = get_media_model()

    # A. News input
    media_news = st.text_area(
        "Enter Digital Media news article:",
        height=150,
        key="media_news"
    )

    # B. Analyze & Predict button
    if st.button("Analyze & Predict Media"):
        # 1) LLM sentiment + text features
        media_features = analyze_media_features(media_news)
        st.subheader("Extracted Features")
        st.write(media_features)

        # 2) Predict returns for all Media tickers
        media_tickers = [
               "NFLX", "DIS", "WBD", "PARA", "SPOT", "RBLX", "EA", "TTWO",
                "SONY", "MTCH", "LYV", "HUYA", "BILI", "TME", "IMAX",
                "MSGE", "PLTK", "CARG", "AMC"
        ]
        media_df_pred = predict_media_returns(model_media, media_features, media_tickers)

        # 3) Store in session state
        st.session_state["media_df_pred"] = media_df_pred
        # (media_news is auto‐stored by widget)

        # 4) Download button
        csv_media = media_df_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Media predictions CSV",
            csv_media,
            "media_predictions.csv",
            "text/csv"
        )

    # C. Plot returns + prediction if available
    if "media_df_pred" in st.session_state:
        media_df_pred = st.session_state["media_df_pred"]
        st.subheader("Predicted Digital Media Returns")
        st.table(media_df_pred[["ticker", "predicted_return"]])

        st.subheader("Return Timeline for Selected Media Stock")
        selected_media = st.selectbox(
            "Choose a stock to visualize:",
            media_df_pred["ticker"].tolist(),
            key="media_plot_selector"
        )
        returns_media, ref_date_media = get_history_returns(selected_media, days=14)
        pred_val_media = float(
            media_df_pred.loc[media_df_pred["ticker"] == selected_media, "predicted_return"]
        )
        fig_media = plot_returns_with_prediction(
            returns_media, pred_val_media, selected_media, ref_date_media
        )
        st.plotly_chart(fig_media, use_container_width=True)
    
    # D. Event Commentary with sector metrics and LLM deep dive
    st.subheader("Event Commentary")

    if "media_df_pred" in st.session_state:
        media_news     = st.session_state.get("media_news", "")
        media_df_pred  = st.session_state["media_df_pred"]

        # Sector‑level metrics
        avg_ret_media = media_df_pred["predicted_return"].mean() * 100
        pos_cnt_media = (media_df_pred["predicted_return"] > 0).sum()
        neg_cnt_media = (media_df_pred["predicted_return"] < 0).sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Sector Avg Return", f"{avg_ret_media:+.2f}%")
        m2.metric("Stocks ↑", pos_cnt_media)
        m3.metric("Stocks ↓", neg_cnt_media)
        st.markdown("---")

        if media_news and st.button("Generate Media Commentary"):
            # Format predictions for the LLM prompt
            media_pred_lines = "\n".join(
                f"- {row.ticker}: {row.predicted_return * 100:+.2f}%"
                for row in media_df_pred.itertuples()
            )
            with st.spinner("Generating detailed Digital Media impact analysis…"):
                media_commentary = llm_media_analysis(media_news, media_pred_lines)

            st.subheader("Detailed Digital Media Impact Commentary")
            st.write(media_commentary)
    else:
        st.info("Please click 'Analyze & Predict Media' above to generate predictions first.")
