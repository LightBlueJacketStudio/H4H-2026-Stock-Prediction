import pandas as pd
from . import feature_engineer as fe
from .train import train_quantile_models
from .evaluate import evaluate_interval

price_df = pd.read_parquet("data/raw_ohlcv.parquet")
news_df  = pd.read_parquet("data/news_daily.parquet")

df = fe.build_features(price_df, news_df)

(models, X_test, y_test, close_test) = train_quantile_models(df, split_date="2024-01-01")
coverage, width, miss_penalty = evaluate_interval(models, X_test, y_test, close_test)

print("Coverage (true in range):", coverage)
print("Avg $ width:", width)
print("Miss penalty:", miss_penalty)