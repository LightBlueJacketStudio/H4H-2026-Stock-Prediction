from models import model as mdl
import pandas as pd

def train_model(df, split_date="2024-01-01"):
    df = df.copy()

    # 1) Ensure we have a real date column
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column for time split.")
    df["date"] = pd.to_datetime(df["date"])

    # 2) Sort by time (critical for time series)
    df = df.sort_values("date").reset_index(drop=True)

    # 3) Split
    split_dt = pd.to_datetime(split_date)
    train_mask = df["date"] < split_dt
    test_mask  = df["date"] >= split_dt

    # 4) Safety checks
    if train_mask.sum() == 0:
        raise ValueError("Train set is empty. Check split_date or date range.")
    if test_mask.sum() == 0:
        raise ValueError("Test set is empty. Check split_date or date range.")

    # 5) Build X/y
    X = df.drop(columns=["y"])
    y = df["y"]

    # drop date from features (usually)
    if "date" in X.columns:
        X = X.drop(columns=["date"])

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # 6) Print “proof” of what data is used
    print(f"Split date: {split_dt.date()}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Train date range: {df.loc[train_mask, 'date'].min().date()} → {df.loc[train_mask, 'date'].max().date()}")
    print(f"Test  date range: {df.loc[test_mask,  'date'].min().date()} → {df.loc[test_mask,  'date'].max().date()}")
    print("Train feature columns:", X_train.columns.tolist()[:10], "...", f"({X_train.shape[1]} total)")

    # 7) Fit
    model = mdl.get_model()
    model.fit(X_train, y_train)

    # 8) Optional: quick sanity score to verify fit actually changed something
    if hasattr(model, "score"):
        train_score = model.score(X_train, y_train)
        print("Train score (sanity):", train_score)

    return model, X_test, y_test