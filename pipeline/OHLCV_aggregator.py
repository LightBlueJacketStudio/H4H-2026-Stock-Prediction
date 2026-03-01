import yfinance as yf
import pandas as pd
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
SYMBOL = "AMZN"
START_DATE = "2020-01-01"
END_DATE = None   # None = today
INTERVAL = "1d"   # daily
OUTPUT_PATH = Path("./data/raw_ohlcv.parquet")


def fetch_ohlcv(symbol: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    print(f"Fetching {symbol} data from yfinance...")

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",  # helps keep columns consistent
    )

    if df.empty:
        raise ValueError("No data downloaded.")

    # ✅ Keep behavior from your original snippet:
    # yfinance sometimes returns MultiIndex columns (e.g., when multiple tickers or some settings)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to make Date a column (same as your snippet)
    df = df.reset_index()

    # Standardize schema
    df = df.rename(columns={"Date": "date"})

    # Ensure datetime + sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def validate_ohlcv(df: pd.DataFrame) -> None:
    required_cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required_cols].isnull().any().any():
        print("Warning: NaN values detected in required columns.")

    print("Validation passed.")


def main():
    df = fetch_ohlcv(SYMBOL, START_DATE, END_DATE, INTERVAL)
    validate_ohlcv(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved OHLCV data to {OUTPUT_PATH}")
    print(df.tail())


if __name__ == "__main__":
    main()