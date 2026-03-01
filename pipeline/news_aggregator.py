import os
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv
import pandas as pd

# Finnhub client
import finnhub

# Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SYMBOL = "AMZN"
DATA_DIR = Path("./data")
#DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = DATA_DIR / "news_raw.parquet"
DAILY_OUT = DATA_DIR / "news_daily.parquet"

# Choose a window that fits your API limits (hack-safe: 180-365 days)
DAYS_BACK = 365
CHUNK_DAYS = 30  # fetch in chunks to avoid response size/rate issues


def daterange_chunks(start: date, end: date, chunk_days: int):
    """Yield (from_date, to_date) pairs covering [start, end] in chunk_days windows."""
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def fetch_finnhub_news(symbol: str, start: date, end: date) -> pd.DataFrame:
    #boostrap
    load_dotenv()  # Load variables from .env file
    
    api_key = os.getenv("FINHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FINNHUB_API_KEY env var")

    client = finnhub.Client(api_key=api_key)

    rows = []
    for f, t in daterange_chunks(start, end, CHUNK_DAYS):
        news = client.company_news(symbol, _from=f.isoformat(), to=t.isoformat())
        rows.extend(news)

    if not rows:
        return pd.DataFrame(columns=["id", "headline", "source", "url", "datetime", "summary"])

    df = pd.DataFrame(rows)

    # Normalize columns (Finnhub returns at least these typically)
    keep = [c for c in ["id", "headline", "source", "url", "datetime", "summary", "category"] if c in df.columns]
    df = df[keep].copy()

    # Finnhub datetime is Unix seconds
    df["published_at"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    df["date"] = df["published_at"].dt.date

    # Drop duplicates by URL/headline/time
    dedup_cols = [c for c in ["url", "headline", "published_at"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    return df


def add_sentiment(df_raw: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    def score(text: str) -> float:
        text = text or ""
        return analyzer.polarity_scores(text)["compound"]

    df = df_raw.copy()
    df["sentiment"] = df["headline"].astype(str).apply(score)

    # Simple bucket for counts
    df["sent_label"] = "neutral"
    df.loc[df["sentiment"] > 0.05, "sent_label"] = "pos"
    df.loc[df["sentiment"] < -0.05, "sent_label"] = "neg"
    return df


def aggregate_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "news_count", "avg_sentiment", "pos_count", "neg_count", "neutral_count", "source_count"])

    daily = (
        df.groupby("date")
          .agg(
              news_count=("headline", "count"),
              avg_sentiment=("sentiment", "mean"),
              pos_count=("sent_label", lambda x: (x == "pos").sum()),
              neg_count=("sent_label", lambda x: (x == "neg").sum()),
              neutral_count=("sent_label", lambda x: (x == "neutral").sum()),
              source_count=("source", "nunique") if "source" in df.columns else ("headline", "count"),
          )
          .reset_index()
    )

    # Convert date to datetime64 for easier merge with OHLCV
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def main():
    end = date.today()
    start = end - timedelta(days=DAYS_BACK)

    print(f"Fetching {SYMBOL} news from {start} to {end}...")
    df_raw = fetch_finnhub_news(SYMBOL, start, end)
    print(f"Fetched {len(df_raw)} articles")

    df_raw = add_sentiment(df_raw)

    # Save raw
    df_raw.to_parquet(RAW_OUT, index=False)
    print(f"Saved raw news to: {RAW_OUT}")

    # Daily aggregate
    df_daily = aggregate_daily(df_raw)
    df_daily.to_parquet(DAILY_OUT, index=False)
    print(f"Saved daily features to: {DAILY_OUT}")

    print(df_daily.tail(10))


if __name__ == "__main__":
    main()