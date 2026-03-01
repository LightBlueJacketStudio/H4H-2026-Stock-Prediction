from feature_engineer import build_features
import pandas as pd

# ----------------------------
# 1) Load raw data
# ----------------------------
price_df = pd.read_parquet("data/raw_ohlcv.parquet")
news_df  = pd.read_parquet("data/news_daily.parquet")

# ----------------------------
# 2) Validate expected schema
# ----------------------------
required_price_cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
missing = [c for c in required_price_cols if c not in price_df.columns]
if missing:
    raise ValueError(f"raw_ohlcv.parquet missing columns: {missing}")

# ensure datetime + sorted (nice for rolling features)
price_df["date"] = pd.to_datetime(price_df["date"])
price_df = price_df.sort_values("date").reset_index(drop=True)

# (optional) ensure news has date too
if "date" in news_df.columns:
    news_df["date"] = pd.to_datetime(news_df["date"])
    news_df = news_df.sort_values("date").reset_index(drop=True)

print("Price columns:", list(price_df.columns))
print(price_df.head())

# ----------------------------
# 3) Build features
# ----------------------------
df_feat = build_features(price_df, news_df)

# ----------------------------
# 4) Split into X and y
# ----------------------------
if "y" not in df_feat.columns:
    raise ValueError("Expected target column 'y' not found in feature dataframe.")

X = df_feat.drop(columns=["date", "y"], errors="ignore")
y = df_feat["y"]

print(df_feat.shape)
print(df_feat.head())
# 1) basic price sanity
print(price_df[["date","Open","High","Low","Close","Adj Close","Volume"]].describe())

# 2) ensure OHLC relationship holds (Low <= Open/Close <= High)
bad = price_df[
    (price_df["Low"] > price_df[["Open","Close"]].min(axis=1)) |
    (price_df["High"] < price_df[["Open","Close"]].max(axis=1))
]
print("Bad OHLC rows:", len(bad))

# 3) date continuity (trading days)
print(price_df["date"].min(), price_df["date"].max(), len(price_df))