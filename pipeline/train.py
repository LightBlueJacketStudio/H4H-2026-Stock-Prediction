#train.py
from models.model import get_model
import pandas as pd

def train_quantile_models(df, split_date="2024-01-01"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    split_dt = pd.to_datetime(split_date)
    train_mask = df["date"] < split_dt
    test_mask  = df["date"] >= split_dt

    # X and y
    y = df["y_ret"]
    X = df.drop(columns=["y_ret"])

    # keep Close for later, drop date from model inputs
    close_test = df.loc[test_mask, "Close"].copy()

    X = X.drop(columns=["date"], errors="ignore")

    # avoid spaces in column names
    X.columns = X.columns.str.replace(" ", "_")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Train/Test:", len(X_train), len(X_test))

    m10 = get_model(objective="quantile", alpha=0.10)
    m90 = get_model(objective="quantile", alpha=0.90)

    m10.fit(X_train, y_train)
    m90.fit(X_train, y_train)

    return (m10, m90), X_test, y_test, close_test