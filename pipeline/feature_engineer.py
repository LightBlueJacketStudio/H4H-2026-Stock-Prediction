#feature_engineer.py
import pandas as pd
import numpy as np

def build_features(
    price_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lags=(1, 2, 3, 5, 10),
    rolling_windows=(5, 10, 20),
) -> pd.DataFrame:
    """
    Builds ML-ready dataset for predicting tomorrow's CLOSE RANGE via quantile models.

    Output includes:
      - engineered features
      - target: y_ret (tomorrow close-to-close return)
      - includes today's Close (needed to convert predicted return quantiles to $ bounds)
      - no leakage: news shifted by 1 day
    """

    # ---------------------------
    # Copy / clean / sort
    # ---------------------------
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Standardize column names for LightGBM (no spaces)
    df.columns = df.columns.str.replace(" ", "_")

    news = news_df.copy()
    news["date"] = pd.to_datetime(news["date"])
    news = news.sort_values("date").reset_index(drop=True)

    # Standardize news column names too
    news.columns = news.columns.str.replace(" ", "_")

    # ---------------------------
    # 1) Returns + TARGET
    # ---------------------------
    df["ret1"] = df["Close"].pct_change()
    # Volatility regime features
    df["rv_5"]  = df["ret1"].rolling(5).std()
    df["rv_20"] = df["ret1"].rolling(20).std()
    df["rv_60"] = df["ret1"].rolling(60).std()

    # Tail / jump indicators
    df["absret_max_20"] = df["ret1"].abs().rolling(20).max()
    df["absret_max_60"] = df["ret1"].abs().rolling(60).max()

    # EWMA vol (often very predictive for next-day range)
    df["ewm_vol_20"] = df["ret1"].ewm(span=20, adjust=False).std()

    # Recent shock (yesterday absolute return)
    df["absret_1"] = df["ret1"].abs()
    # ✅ Target for quantiles: tomorrow return
    # r_{t+1} = Close_{t+1}/Close_t - 1
    df["y_ret"] = np.log(df["Close"].shift(-1) / df["Close"])

    # ---------------------------
    # 2) Lag features
    # ---------------------------
    for lag in lags:
        df[f"ret_lag_{lag}"] = df["ret1"].shift(lag)
        df[f"vol_lag_{lag}"] = df["Volume"].shift(lag)
        df[f"absret_lag_{lag}"] = df["ret1"].abs().shift(lag)

    # ---------------------------
    # 3) Rolling stats (volatility-aware)
    # ---------------------------
    for window in rolling_windows:
        df[f"ret_mean_{window}"] = df["ret1"].rolling(window).mean()
        df[f"ret_std_{window}"]  = df["ret1"].rolling(window).std()
        df[f"absret_mean_{window}"] = df["ret1"].abs().rolling(window).mean()
        df[f"vol_mean_{window}"] = df["Volume"].rolling(window).mean()

    # Moving averages + ratios
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["close_ma5_ratio"] = df["Close"] / df["ma_5"]
    df["close_ma20_ratio"] = df["Close"] / df["ma_20"]

    # Intraday range / ATR-ish (vol proxy)
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["hl_spread_14"] = df["hl_spread"].rolling(14).mean()

    # ---------------------------
    # 4) Merge news (shift to avoid leakage)
    # ---------------------------
    df = df.merge(news, on="date", how="left")

    news_cols = [c for c in news.columns if c != "date"]
    for col in news_cols:
        df[col] = df[col].fillna(0)

    # shift news by 1 day so today's features use yesterday's news
    df[news_cols] = df[news_cols].shift(1)

    # ---------------------------
    # 5) Calendar features
    # ---------------------------
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # ---------------------------
    # 6) Drop NaNs (created by pct_change, rolls, lags, shift(-1))
    # ---------------------------
    df = df.dropna().reset_index(drop=True)

    return df