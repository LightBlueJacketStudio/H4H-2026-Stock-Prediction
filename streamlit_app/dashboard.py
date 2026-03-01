# streamlit_app/dashboard.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Import feature engineer from pipeline/
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent           # .../streamlit_app
PROJECT_ROOT = BASE_DIR.parent                      # .../H4H-2026-Stock-Prediction

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.feature_engineer import build_features  # uses your pipeline file

# -----------------------------
# Stable absolute paths to data
# -----------------------------
DATA_DIR = PROJECT_ROOT / "data"
OHLCV_FILE = DATA_DIR / "raw_ohlcv.parquet"
NEWS_DAILY_FILE = DATA_DIR / "news_daily.parquet"
NEWS_RAW_FILE = DATA_DIR / "news_raw.parquet"  # optional, if you want raw headlines


@st.cache_data(ttl=60)
def load_parquets():
    """Load parquet data produced by pipeline."""
    if not OHLCV_FILE.exists():
        raise FileNotFoundError(f"Missing: {OHLCV_FILE}")
    if not NEWS_DAILY_FILE.exists():
        raise FileNotFoundError(f"Missing: {NEWS_DAILY_FILE}")

    ohlcv = pd.read_parquet(OHLCV_FILE)
    news_daily = pd.read_parquet(NEWS_DAILY_FILE)

    # Ensure date column exists & is datetime
    if "date" not in ohlcv.columns:
        # in case parquet saved index
        ohlcv = ohlcv.reset_index().rename(columns={"index": "date", "Date": "date"})
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv.sort_values("date").reset_index(drop=True)

    if "date" not in news_daily.columns:
        news_daily = news_daily.reset_index().rename(columns={"index": "date", "Date": "date"})
    news_daily["date"] = pd.to_datetime(news_daily["date"])
    news_daily = news_daily.sort_values("date").reset_index(drop=True)

    return ohlcv, news_daily


def compute_placeholders(ohlcv: pd.DataFrame) -> dict:
    """
    Until your model is fully wired, compute:
    - predicted range from volatility
    - confidence from momentum
    - risk from volatility
    """
    close = ohlcv["Close"].astype(float)

    current = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else current
    delta = current - prev
    delta_pct = (delta / prev * 100) if prev else 0.0

    # volatility (20-day)
    ret = close.pct_change()
    vol20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 20 else float(ret.std())
    vol20 = 0.0 if np.isnan(vol20) else vol20

    band = current * (2.0 * vol20)  # ~2σ range
    pred_low = current - band
    pred_high = current + band

    # momentum-based confidence (5-day return scaled)
    mom5 = float(close.pct_change(5).iloc[-1]) if len(close) > 6 else 0.0
    conf = 0.5 + np.clip(mom5 * 6.0, -0.45, 0.45)
    conf = float(np.clip(conf, 0.05, 0.95))

    # risk from volatility
    risk_score = int(np.clip(vol20 * 1200, 0, 100))
    risk_label = "Low" if risk_score < 35 else "Medium" if risk_score < 70 else "High"

    return {
        "current_price": current,
        "delta": delta,
        "delta_pct": delta_pct,
        "pred_low": pred_low,
        "pred_high": pred_high,
        "confidence": conf,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "vol20": vol20,
        "mom5": mom5,
    }


def render_dashboard(go_home):
    st.set_page_config(page_title="Glimpse Dashboard", layout="wide")

    ticker = st.session_state.get("ticker", "")
    if not ticker:
        st.warning("No ticker selected.")
        if st.button("← Back"):
            go_home()
        return

    # -----------------------------
    # Load real data
    # -----------------------------
    try:
        ohlcv, news_daily = load_parquets()
    except FileNotFoundError as e:
        st.error("Parquet files not found. Run the pipeline scripts first.")
        st.write(str(e))
        st.write("Expected in:", str(DATA_DIR))
        st.write("- raw_ohlcv.parquet")
        st.write("- news_daily.parquet")
        if st.button("← Back"):
            go_home()
        st.stop()

    # Optional: build features (real pipeline step)
    # This produces engineered features + target y inside the dataframe.
    # If build_features fails for any reason, we still show the dashboard.
    feat_df = None
    try:
        feat_df = build_features(ohlcv, news_daily)
    except Exception:
        feat_df = None

    # Placeholders (until model outputs are ready)
    p = compute_placeholders(ohlcv)

    # -----------------------------
    # HEADER ROW: back button (wide) + title
    # -----------------------------
    top_left, top_right = st.columns([2, 10])

    with top_left:
        if st.button("← Back"):
            go_home()

    with top_right:
        st.title("Prediction Dashboard")

    st.divider()

    # -----------------------------
    # ROW 1: Company | Predicted Range | Confidence
    # -----------------------------
    c1, c2, c3 = st.columns([2, 3, 2])

    with c1:
        st.subheader("Company")
        st.metric(
            label=f"{ticker} Current Price",
            value=f"${p['current_price']:.2f}",
            delta=f"{p['delta']:+.2f} ({p['delta_pct']:+.2f}%)"
        )
        st.caption(f"OHLCV rows: {len(ohlcv)} | News days: {len(news_daily)}")

    with c2:
        st.subheader("Predicted Close Price Range")
        st.write(f"### ${p['pred_low']:.2f} – ${p['pred_high']:.2f}")
        st.caption("Currently placeholder from recent volatility (swap with model output).")

    with c3:
        st.subheader("Confidence")
        st.metric("P(Up)", f"{p['confidence']*100:.0f}%")
        st.caption("Currently placeholder from short-term momentum (swap with model output).")

    st.divider()

    # -----------------------------
    # ROW 2: Chart | Risk | News
    # -----------------------------
    r2_left, r2_mid, r2_right = st.columns([4, 1.6, 3])

    with r2_left:
        st.subheader("Line Graph: Past Predicted Price Range vs Actual Price")

        chart = ohlcv[["date", "Close"]].tail(90).copy()
        chart = chart.rename(columns={"Close": "actual_close"})
        # constant placeholder band for now; later replace with per-day predicted bands
        chart["pred_low"] = p["pred_low"]
        chart["pred_high"] = p["pred_high"]
        chart = chart.set_index("date")

        st.line_chart(chart[["actual_close", "pred_low", "pred_high"]])
        st.caption("Later: use Altair/Plotly for a shaded prediction band.")

    with r2_mid:
        st.subheader("Risk Level")
        st.metric("Indicator", p["risk_label"])
        st.progress(p["risk_score"] / 100.0)
        st.caption(f"Risk Score: {p['risk_score']}/100")

    with r2_right:
        st.subheader("News (Daily Aggregates)")
        cols = [c for c in
                ["date", "news_count", "avg_sentiment", "pos_count", "neg_count", "neutral_count", "source_count"]
                if c in news_daily.columns]
        st.dataframe(news_daily[cols].tail(12), use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------
    # ROW 3: Past Prediction table | Helpful Info | Actions
    # -----------------------------
    r3_left, r3_mid, r3_right = st.columns([4, 4, 2])

    with r3_left:
        st.subheader("Past Prediction w/ News")

        # If you later store prediction history, replace this section.
        if feat_df is None:
            st.info("Prediction history will appear here after the model is connected.")
        else:
            # show last engineered rows as proof features exist
            preview_cols = [c for c in ["date", "Close", "ret_1", "vol_20", "avg_sentiment_lag1", "y"] if c in feat_df.columns]
            st.dataframe(feat_df[preview_cols].tail(10), use_container_width=True, hide_index=True)

    with r3_mid:
        st.subheader("Helpful Information")
        st.write(f"- **Model:** Gradient Boosting (pipeline ready; hook inference next)")
        st.write(f"- **Volatility (20d):** {p['vol20']:.4f}")
        st.write(f"- **Momentum (5d):** {p['mom5']:.4f}")
        st.write("- **Indices:** Nasdaq / Dow / S&P500 (add later if you want)")
        st.write("- **Data Sources:** yfinance OHLCV + Finnhub headlines → sentiment aggregates")

    with r3_right:
        st.subheader("Actions")
        st.button("Add to Watch list", use_container_width=True)
        st.button("Report Inaccuracy", use_container_width=True)
        with st.expander("How we get our prediction"):
            st.write(
                """
**High level flow:**
- Pull OHLCV (Open/High/Low/Close/Adj Close/Volume) → `raw_ohlcv.parquet`
- Pull Finnhub headlines → sentiment scoring → daily aggregates → `news_daily.parquet`
- Feature engineering (returns, lags, rolling stats, MA ratios + shifted news)
- Train Gradient Boosting with time-series split
- Output next-day direction probability (Confidence) + predicted range
                """
            )