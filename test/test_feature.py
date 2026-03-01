from pipeline import feature_engineer as fe

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
df_feat = fe.build_features(price_df, news_df)

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


df_feat = fe.build_features(price_df, news_df)

print(df_feat.shape)
print(df_feat.columns.tolist())

# no NaNs
assert df_feat.isna().sum().sum() == 0

# required columns exist
required = ["date", "Close", "Volume", "ret1", "y", "ret_lag_1", "ret_mean_5", "news_count", "avg_sentiment"]
missing = [c for c in required if c not in df_feat.columns]
assert not missing, f"Missing columns: {missing}"

# dates strictly increasing
assert df_feat["date"].is_monotonic_increasing

i = 50  # any index safely away from start/end
row = df_feat.iloc[i]

# Find same date in original price_df
d = row["date"]
orig_idx = price_df.index[price_df["date"] == d][0]

close_t   = price_df.loc[orig_idx, "Close"]
close_tp1 = price_df.loc[orig_idx + 1, "Close"]

true_next_ret = (close_tp1 / close_t) - 1

print("Feature y:", row["y"])
print("True next ret:", true_next_ret)

# allow tiny floating error
assert abs(row["y"] - true_next_ret) < 1e-10

i = 50
row = df_feat.iloc[i]

# ret_lag_1 should equal ret1 from previous row in df_feat
prev_row = df_feat.iloc[i - 1]

print("ret_lag_1:", row["ret_lag_1"])
print("prev ret1:", prev_row["ret1"])

assert abs(row["ret_lag_1"] - prev_row["ret1"]) < 1e-12

lag = 5
i = 50
row = df_feat.iloc[i]
lag_row = df_feat.iloc[i - lag]

assert abs(row[f"ret_lag_{lag}"] - lag_row["ret1"]) < 1e-12

i = 50
window = 5

manual = df_feat["ret1"].iloc[i-window+1:i+1].mean()
computed = df_feat[f"ret_mean_{window}"].iloc[i]

print("manual mean:", manual)
print("computed:", computed)

assert abs(manual - computed) < 1e-12

manual_std = df_feat["ret1"].iloc[i-window+1:i+1].std()
computed_std = df_feat[f"ret_std_{window}"].iloc[i]
assert abs(manual_std - computed_std) < 1e-12

news_cols = [c for c in news_df.columns if c != "date"]

print(df_feat[news_cols].describe())
# At least one day should have news_count > 0 if your news_df has coverage

# pick a date where you know there was news
d = df_feat["date"].iloc[50]

news_today = news_df.loc[news_df["date"] == d, "news_count"]
news_yday  = news_df.loc[news_df["date"] == (d - pd.Timedelta(days=1)), "news_count"]

feat_news = df_feat.loc[df_feat["date"] == d, "news_count"].iloc[0]

print("feature news_count on d:", feat_news)
print("raw news_count on d:", float(news_today.iloc[0]) if len(news_today) else None)
print("raw news_count on d-1:", float(news_yday.iloc[0]) if len(news_yday) else None)

# If yesterday exists in news_df, your feature should match yesterday.
if len(news_yday):
    assert abs(feat_news - float(news_yday.iloc[0])) < 1e-9

print(df_feat["ret1"].describe())
# Typical daily stock returns are usually within a few percent.
# If you see returns like 2.5 or -0.9 often, something is wrong (bad scaling or wrong column).