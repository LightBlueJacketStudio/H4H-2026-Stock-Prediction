from . import feature_engineer as fe
from . import train
#from models import model
from . import evaluate
import pandas as pd

price_df = pd.read_parquet("data/raw_ohlcv.parquet")
news_df  = pd.read_parquet("data/news_daily.parquet")
df = fe.build_features(price_df, news_df)

model, X_test, y_test = train.train_model(df)
rmse, direction = evaluate.evaluate(model, X_test, y_test)

print("RMSE:", rmse)
print("Directional Accuracy:", direction)

ticker = "AMZN"  # or pass in
model, model_path, X_test, y_test = train.train_and_save(df, ticker=ticker)
print("Saved model to:", model_path)