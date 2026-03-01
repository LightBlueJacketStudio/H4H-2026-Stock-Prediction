import pandas as pd
from . import feature_engineer as fe
from .train import train_quantile_models
from .evaluate import evaluate_interval
ticker = "AMZN"
price_df = pd.read_parquet("data/raw_ohlcv.parquet")
news_df  = pd.read_parquet("data/news_daily.parquet")

df = fe.build_features(price_df, news_df)

(models, X_test, y_test, close_test) = train_quantile_models(df, split_date="2024-01-01")

# ======================
# DEBUG SECTION
# ======================

m10, m90 = models

print("Index aligned X/y:", X_test.index.equals(y_test.index))
print("Index aligned X/close:", X_test.index.equals(close_test.index))

print("y_test (returns) sample:", y_test.head().tolist())
print("close_test sample:", close_test.head().tolist())

r10 = m10.predict(X_test)
r90 = m90.predict(X_test)

print("Crossing rate:", (r10 > r90).mean())
print("Pred return q10/q90 sample:", list(zip(r10[:5], r90[:5])))

# ======================

coverage, width, miss_penalty = evaluate_interval(models, X_test, y_test, close_test)

print("Coverage (true in range):", coverage)
print("Avg $ width:", width)
print("Miss penalty:", miss_penalty)

print("creating joblib")
import joblib

(models, X_test, y_test, close_test) = train_quantile_models(df, split_date="2024-01-01")

m10, m90 = models

joblib.dump(
    {
        "model_q10": m10,
        "model_q90": m90,
    },
    f"models/{ticker}_model.joblib"
)

print(f"Model saved to models/{ticker}_model.joblib")