# train.py
from models.model import get_model
import lightgbm as lgb
import pandas as pd

def train_quantile_models(df, split_date="2024-01-01"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    split_dt = pd.to_datetime(split_date)
    train_mask = df["date"] < split_dt
    test_mask  = df["date"] >= split_dt

    # Target
    y = df["y_ret"]

    # Save Close for conversion later (only for test period)
    close_test = df.loc[test_mask, "Close"].copy()

    # Features: drop target/date; optionally drop Close to avoid "cheating"
    X = df.drop(columns=["y_ret", "date"], errors="ignore")
    X = X.drop(columns=["Close"], errors="ignore")  # recommended for clean separation

    # avoid spaces in column names
    X.columns = X.columns.str.replace(" ", "_")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Train/Test:", len(X_train), len(X_test))

    # after X_train/y_train created:
    n = len(X_train)
    val_start = int(n * 0.85)

    X_tr, X_val = X_train.iloc[:val_start], X_train.iloc[val_start:]
    y_tr, y_val = y_train.iloc[:val_start], y_train.iloc[val_start:]

    m10 = get_model(objective="quantile", alpha=0.10)
    m90 = get_model(objective="quantile", alpha=0.90)

    m10.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    m90.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    return (m10, m90), X_test, y_test, close_test