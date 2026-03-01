# pipeline/feature_engineering.py

import pandas as pd
import numpy as np


def build_features(price_df: pd.DataFrame,
                   news_df: pd.DataFrame,
                   lags=(1, 2, 3, 5, 10),
                   rolling_windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Build ML-ready dataset from OHLCV + daily news features.

    Inputs
    -------
    price_df : DataFrame
        Must contain columns:
        ['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    news_df : DataFrame
        Must contain:
        ['date', 'news_count', 'avg_sentiment', ...]

    Returns
    -------
    DataFrame with:
        - engineered features
        - target column 'y'
        - no NaNs
    """

    # ---------------------------
    # Copy and sort
    # ---------------------------
    df = price_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])
    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"])

    # ---------------------------
    # 1️⃣ Returns
    # ---------------------------
    df["ret1"] = df["Close"].pct_change()

    # Target = next-day return
    df["y"] = df["ret1"].shift(-1)

    # ---------------------------
    # 2️⃣ Lag Features
    # ---------------------------
    for lag in lags:
        df[f"ret_lag_{lag}"] = df["ret1"].shift(lag)
        df[f"vol_lag_{lag}"] = df["Volume"].shift(lag)

    # ---------------------------
    # 3️⃣ Rolling Statistics
    # ---------------------------
    for window in rolling_windows:
        df[f"ret_mean_{window}"] = df["ret1"].rolling(window).mean()
        df[f"ret_std_{window}"] = df["ret1"].rolling(window).std()
        df[f"vol_mean_{window}"] = df["Volume"].rolling(window).mean()

    # Price-based rolling
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()

    df["close_ma5_ratio"] = df["Close"] / df["ma_5"]
    df["close_ma20_ratio"] = df["Close"] / df["ma_20"]

    # High-low spread (intraday volatility proxy)
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]

    # ---------------------------
    # 4️⃣ Merge News Features
    # ---------------------------
    df = df.merge(news_df, on="date", how="left")

    # Fill missing news (no news day)
    news_cols = [c for c in news_df.columns if c != "date"]
    for col in news_cols:
        df[col] = df[col].fillna(0)

    # IMPORTANT: shift news to avoid leakage
    for col in news_cols:
        df[col] = df[col].shift(1)

    # ---------------------------
    # 5️⃣ Calendar Features (optional but useful)
    # ---------------------------
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # ---------------------------
    # 6️⃣ Drop NaNs
    # ---------------------------
    df = df.dropna().reset_index(drop=True)
    return df