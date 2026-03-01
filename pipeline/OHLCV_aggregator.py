import yfinance as yf
import pandas as pd
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
SYMBOL = "AMZN"
START_DATE = "2020-01-01"
END_DATE = None   # None = today
INTERVAL = "1d"   # 1d for daily (hackathon safe)
OUTPUT_PATH = Path("./data/raw_ohlcv.parquet")


def fetch_ohlcv(symbol, start, end=None, interval="1d"):
    print(f"Fetching {symbol} data from yfinance...")
    
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded.")

    # Reset index to make Date a column
    df.reset_index(inplace=True)

    # Rename to standardized schema
    df.rename(columns={
        "Date": "date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume"
    }, inplace=True)

    # Ensure datetime type
    df["date"] = pd.to_datetime(df["date"])

    # Sort just in case
    df = df.sort_values("date").reset_index(drop=True)

    return df


def validate_ohlcv(df):
    required_cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df.isnull().sum().sum() > 0:
        print("Warning: NaN values detected.")

    print("Validation passed.")


def main():
    df = fetch_ohlcv(SYMBOL, START_DATE, END_DATE, INTERVAL)

    validate_ohlcv(df)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved OHLCV data to {OUTPUT_PATH}")
    print(df.tail())


if __name__ == "__main__":
    main()