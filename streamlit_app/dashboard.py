# streamlit_app/dashboard.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# Import feature engineer from pipeline/
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent           # .../streamlit_app
PROJECT_ROOT = BASE_DIR.parent                      # .../H4H-2026-Stock-Prediction

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.feature_engineer import build_features  # builds 'y' and feature columns

# -----------------------------
# Stable absolute paths to data/model
# -----------------------------
DATA_DIR = PROJECT_ROOT / "data"
OHLCV_FILE = DATA_DIR / "raw_ohlcv.parquet"
NEWS_DAILY_FILE = DATA_DIR / "news_daily.parquet"

MODELS_DIR = PROJECT_ROOT / "models"


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


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def load_model_for_ticker(ticker: str):
    """Load models/{ticker}_model.joblib if it exists."""
    model_path = MODELS_DIR / f"{ticker}_model.joblib"
    if not model_path.exists():
        return None, model_path
    try:
        return joblib.load(model_path), model_path
    except Exception:
        return None, model_path


def align_X_to_model(model_bundle, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align to training columns (prefer feature_columns saved in joblib),
    otherwise fall back to feature_names_in_ on one of the models.
    """
    # model_bundle is a dict: {"model_q10":..., "model_q90":..., "feature_columns":[...]}
    cols = None

    if isinstance(model_bundle, dict) and "feature_columns" in model_bundle:
        cols = list(model_bundle["feature_columns"])
    else:
        # fallback: use feature_names_in_ from q10 if available
        m = model_bundle["model_q10"] if isinstance(model_bundle, dict) else model_bundle
        if hasattr(m, "feature_names_in_"):
            cols = list(m.feature_names_in_)

    if cols is not None:
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols]

    return X


def compute_real_outputs(ticker: str, ohlcv: pd.DataFrame, news_daily: pd.DataFrame):
    """
    Uses:
      - build_features() output (predicts next-day return y)
      - saved model models/{ticker}_model.joblib

    Returns:
      p dict for UI, feat_df for preview, model for chart computation (optional)
    """
    feat_df = build_features(ohlcv, news_daily)

    model, model_path = load_model_for_ticker(ticker)
    if model is None:
        return None, feat_df, model_path, None

    # Latest row inference (predict next-day return)
    latest = feat_df.iloc[[-1]].copy()
    X_latest = latest.drop(columns=["y_ret", "date"], errors="ignore")
    X_latest = align_X_to_model(model, X_latest)

    #Amy's code
    # pred_ret = float(model.predict(X_latest)[0])  # next-day return prediction

    # # Current + delta from OHLCV (true market values)
    # close = ohlcv["Close"].astype(float)

    # current = float(close.iloc[-1])
    # prev = float(close.iloc[-2]) if len(close) > 1 else current
    # delta = current - prev
    # delta_pct = (delta / prev * 100) if prev else 0.0

    # # Convert predicted return -> predicted close (center)
    # pred_close = current * (1.0 + pred_ret)

    # # Use a volatility estimate for a range band.
    # # Prefer engineered rolling std if present (ret_std_20), else compute quickly.
    # if "ret_std_20" in feat_df.columns:
    #     vol20 = float(feat_df["ret_std_20"].iloc[-1])
    # else:
    #     vol20 = float(close.pct_change().rolling(20).std().iloc[-1])

    # if np.isnan(vol20):
    #     vol20 = 0.0

    # band = pred_close * (2.0 * vol20)  # 2-sigma band
    # pred_low = pred_close - band
    # pred_high = pred_close + band

    # Confidence (regression -> proxy)
    # Stronger predicted return relative to volatility => higher confidence.

    #sean's tweak
    bundle, model_path = load_model_for_ticker(ticker)
    if bundle is None:
        return None, feat_df, model_path, None

    m10 = bundle["model_q10"]
    m90 = bundle["model_q90"]

    latest = feat_df.iloc[[-1]].copy()
    X_latest = latest.drop(columns=["y_ret", "date"], errors="ignore")
    X_latest = align_X_to_model(bundle, X_latest)

    r10 = float(m10.predict(X_latest)[0])
    r90 = float(m90.predict(X_latest)[0])

    # ensure ordering
    r_low, r_high = (min(r10, r90), max(r10, r90))
    close_series = ohlcv["Close"].astype(float)
    current = float(close_series.iloc[-1])
    prev = float(close_series.iloc[-2]) if len(close_series) > 1 else current
    delta = current - prev
    delta_pct = (delta / prev * 100) if prev else 0.0

    # log-return -> price bounds
    pred_low = current * np.exp(r_low)
    pred_high = current * np.exp(r_high)

    # optional center for display
    pred_close = current * np.exp(0.5 * (r_low + r_high))
    pred_ret = 0.5 * (r_low + r_high)

    #amy
    # z = pred_ret / (vol20 + 1e-6)
    # conf = float(np.clip(sigmoid(z), 0.05, 0.95))

    # # Risk from volatility
    # risk_score = int(np.clip(vol20 * 1200, 0, 100))
    # risk_label = "Low" if risk_score < 35 else "Medium" if risk_score < 70 else "High"

    # # Momentum (5d) for helpful info
    # mom5 = float(close.pct_change(5).iloc[-1]) if len(close) > 6 else 0.0
    # if np.isnan(mom5):
    #     mom5 = 0.0
        # Volatility (20d)

    #sean
    if "ret_std_20" in feat_df.columns:
        vol20 = float(feat_df["ret_std_20"].iloc[-1])
    else:
        vol20 = float(close_series.pct_change().rolling(20).std().iloc[-1])
    if np.isnan(vol20):
        vol20 = 0.0

    # Confidence (recommended): narrower predicted band => higher confidence
    interval_width = float(pred_high - pred_low)
    width_ratio = interval_width / (current + 1e-6)
    conf = float(np.clip(1.0 - width_ratio * 5.0, 0.05, 0.95))

    # If you prefer your old z-score confidence, use this instead:
    # z = pred_ret / (vol20 + 1e-6)
    # conf = float(np.clip(sigmoid(z), 0.05, 0.95))

    # Risk from volatility
    risk_score = int(np.clip(vol20 * 1200, 0, 100))
    risk_label = "Low" if risk_score < 35 else "Medium" if risk_score < 70 else "High"

    # Momentum (5d)
    mom5 = float(close_series.pct_change(5).iloc[-1]) if len(close_series) > 6 else 0.0
    if np.isnan(mom5):
        mom5 = 0.0

    p = {
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
        "pred_ret": pred_ret,
        "pred_close": pred_close,
        "model_name": type(model).__name__,
    }

    return p, feat_df, model_path, model

def AMY_build_prediction_history(feat_df, model, lookback=15):
    """
    Build historical prediction vs actual comparison table.
    Since y is next-day return, align predictions to next day's close.
    """
    df = feat_df.sort_values("date").reset_index(drop=True)

    if len(df) < 10:
        return pd.DataFrame()

    # Use last N rows for display
    df = df.tail(lookback + 1).reset_index(drop=True)

    X = df.drop(columns=["y_ret", "date"], errors="ignore")

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    pred_ret = model.predict(X)

    # predicted close at time t -> actual at t+1
    pred_close = df["Close"].values * (1 + pred_ret)

    # Align next-day
    table = pd.DataFrame({
        "date": df["date"].iloc[1:].values,
        "Predicted Close": pred_close[:-1],
        "Actual Close": df["Close"].iloc[1:].values,
    })

    table["% Error"] = (
        (table["Predicted Close"] - table["Actual Close"])
        / table["Actual Close"]
        * 100
    )

    table = table.sort_values("date", ascending=False).reset_index(drop=True)

    return table.round({
        "Predicted Close": 2,
        "Actual Close": 2,
        "% Error": 2
    })
def AMY_build_model_band_chart(feat_df: pd.DataFrame, model, lookback: int = 140) -> pd.DataFrame:
    """
    Build a true *model-based* historical prediction band.

    Since target y is next-day return (shift -1), prediction at time t corresponds to close at t+1.
    We align:
      - pred_close_t -> actual_close_{t+1}
      - date_{t+1} used as the chart x-axis
    """
    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 5:
        return pd.DataFrame()

    df = df.tail(lookback).reset_index(drop=True)

    # X for all rows
    X_all = df.drop(columns=["y_ret", "date"], errors="ignore")
    X_all = align_X_to_model(model, X_all)

    pred_ret = model.predict(X_all).astype(float)

    # Predicted close based on today's close * (1 + pred_ret)
    # Here 'Close' in df is the close at date t
    pred_close = df["Close"].astype(float).values * (1.0 + pred_ret)

    # Volatility per row for band: prefer engineered ret_std_20 if available
    if "ret_std_20" in df.columns:
        vol = df["ret_std_20"].astype(float).values
    else:
        vol = df["Close"].astype(float).pct_change().rolling(20).std().values

    vol = np.where(np.isnan(vol), 0.0, vol)
    band = pred_close * (2.0 * vol)

    pred_low = pred_close - band
    pred_high = pred_close + band

    # Align prediction at t to actual at t+1 (next day)
    out = pd.DataFrame({
        "date": df["date"].values,
        "actual_close_t": df["Close"].astype(float).values,
        "pred_close_t": pred_close,
        "pred_low_t": pred_low,
        "pred_high_t": pred_high,
    })

    # Shift to next day alignment: use next day's date and actual close as comparison
    out = out.iloc[:-1].copy()
    out["date"] = df["date"].iloc[1:].values
    out["actual_close"] = df["Close"].astype(float).iloc[1:].values

    out = out.rename(columns={"pred_low_t": "pred_low", "pred_high_t": "pred_high"})
    out = out[["date", "actual_close", "pred_low", "pred_high"]].dropna()

    return out

def build_prediction_history(feat_df, bundle, lookback=15):
    """
    Historical prediction vs actual comparison table.

    - bundle: {"model_q10":..., "model_q90":..., optional "feature_columns":...}
    - Target is log-return, so price = Close_t * exp(pred_logret)
    - Predictions at time t correspond to actual close at t+1
    """
    df = feat_df.sort_values("date").reset_index(drop=True)
    if len(df) < 10:
        return pd.DataFrame()

    df = df.tail(lookback + 1).reset_index(drop=True)

    m10 = bundle["model_q10"]
    m90 = bundle["model_q90"]

    X = df.drop(columns=["y_ret", "date"], errors="ignore")
    X = align_X_to_model(bundle, X)

    r10 = m10.predict(X).astype(float)
    r90 = m90.predict(X).astype(float)

    r_low = np.minimum(r10, r90)
    r_high = np.maximum(r10, r90)

    # Use midpoint (center log-return) as point prediction
    r_mid = 0.5 * (r_low + r_high)

    # predicted close at time t
    pred_close_t = df["Close"].astype(float).values * np.exp(r_mid)

    # Align to next-day actual close
    table = pd.DataFrame({
        "date": df["date"].iloc[1:].values,
        "Predicted Close": pred_close_t[:-1],
        "Actual Close": df["Close"].astype(float).iloc[1:].values,
    })

    table["% Error"] = (
        (table["Predicted Close"] - table["Actual Close"])
        / table["Actual Close"] * 100
    )

    table = table.sort_values("date", ascending=False).reset_index(drop=True)

    return table.round({
        "Predicted Close": 2,
        "Actual Close": 2,
        "% Error": 2
    })

def build_model_band_chart(feat_df: pd.DataFrame, bundle, lookback: int = 140) -> pd.DataFrame:
    """
    True model-based historical prediction band using quantile models (Q10/Q90).

    Target is log-return:
      pred_low_t  = Close_t * exp(r_low_t)
      pred_high_t = Close_t * exp(r_high_t)

    Align prediction at t to actual close at t+1.
    """
    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 5:
        return pd.DataFrame()

    df = df.tail(lookback).reset_index(drop=True)

    m10 = bundle["model_q10"]
    m90 = bundle["model_q90"]

    X_all = df.drop(columns=["y_ret", "date"], errors="ignore")
    X_all = align_X_to_model(bundle, X_all)

    r10 = m10.predict(X_all).astype(float)
    r90 = m90.predict(X_all).astype(float)

    r_low = np.minimum(r10, r90)
    r_high = np.maximum(r10, r90)

    close_t = df["Close"].astype(float).values

    pred_low_t = close_t * np.exp(r_low)
    pred_high_t = close_t * np.exp(r_high)

    out = pd.DataFrame({
        "date_t": df["date"].values,
        "actual_close_t": close_t,
        "pred_low_t": pred_low_t,
        "pred_high_t": pred_high_t,
    })

    # Align: prediction from t is for t+1
    out = out.iloc[:-1].copy()
    out["date"] = df["date"].iloc[1:].values
    out["actual_close"] = df["Close"].astype(float).iloc[1:].values

    out = out.rename(columns={"pred_low_t": "pred_low", "pred_high_t": "pred_high"})
    out = out[["date", "actual_close", "pred_low", "pred_high"]].dropna()

    return out

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

    # -----------------------------
    # Compute real model outputs (fallback to placeholders if missing model)
    # -----------------------------
    p, feat_df, model_path, model = compute_real_outputs(ticker, ohlcv, news_daily)

    using_placeholders = False
    if p is None:
        using_placeholders = True
        st.warning(f"Using placeholders (no trained model found at: {model_path})")

        # Lightweight placeholder: keep same fields so UI doesn't change
        close = ohlcv["Close"].astype(float)
        current = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else current
        delta = current - prev
        delta_pct = (delta / prev * 100) if prev else 0.0

        vol20 = float(close.pct_change().rolling(20).std().iloc[-1])
        vol20 = 0.0 if np.isnan(vol20) else vol20
        band = current * (2.0 * vol20)
        pred_low = current - band
        pred_high = current + band

        mom5 = float(close.pct_change(5).iloc[-1]) if len(close) > 6 else 0.0
        conf = float(np.clip(0.5 + np.clip(mom5 * 6.0, -0.45, 0.45), 0.05, 0.95))

        risk_score = int(np.clip(vol20 * 1200, 0, 100))
        risk_label = "Low" if risk_score < 35 else "Medium" if risk_score < 70 else "High"

        p = {
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
            "model_name": "Placeholder",
        }

    # -----------------------------
    # HEADER ROW: back button + title (same layout)
    # -----------------------------
    top_left, top_right = st.columns([1, 10])

    with top_left:
        if st.button("← Back"):
            go_home()

    with top_right:
        st.title("Prediction Dashboard")

    st.divider()

    # -----------------------------
    # ROW 1: Company | Predicted Range | Confidence (same layout)
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
        # st.write(f"### ${p['pred_low']:.2f} – ${p['pred_high']:.2f}")
        current = p["current_price"]
        low = p["pred_low"]
        high = p["pred_high"]

        low_color = "#22c55e" if low > current else "#ef4444"
        high_color = "#22c55e" if high > current else "#ef4444"

        st.markdown(
            f"""
            <h2>
                <span style="color:{low_color}; font-weight:600;">
                    ${low:.2f}
                </span>
                –
                <span style="color:{high_color}; font-weight:600;">
                    ${high:.2f}
                </span>
            </h2>
            """,
            unsafe_allow_html=True
        )
        if using_placeholders:
            st.caption("Placeholder range from recent volatility.")
        else:
            st.caption("Model-predicted close centered band (2σ volatility).")

    with c3:
        st.subheader("Confidence")
        st.metric("P(Up)", f"{p['confidence']*100:.0f}%")
        if using_placeholders:
            st.caption("Placeholder confidence from short-term momentum.")
        else:
            st.caption("Regression → confidence proxy via predicted return / volatility.")

    st.divider()

    # -----------------------------
    # ROW 2: Chart | Risk | News
    # -----------------------------
    r2_left, r2_mid, r2_right = st.columns([4, 1.6, 3])

    with r2_left:
        st.subheader("Line Graph: Past Predicted Price Range vs Actual Price")

        if (not using_placeholders) and (feat_df is not None) and (model is not None):
            band_df = build_model_band_chart(feat_df, model, lookback=160)
            if not band_df.empty:
                chart = band_df.tail(120).set_index("date")[["actual_close", "pred_low", "pred_high"]]
                st.line_chart(chart)
                st.caption("Historical band is model-based (aligned to next-day close).")
            else:
                st.info("Not enough data to draw model band. Showing volatility band instead.")
        else:
            # Fallback: volatility band around close (still dynamic)
            hist = ohlcv[["date", "Close"]].copy()
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.sort_values("date").reset_index(drop=True)
            close = hist["Close"].astype(float)
            ret = close.pct_change()
            rolling_vol = ret.rolling(20).std()
            band = close * (2.0 * rolling_vol)
            hist["actual_close"] = close
            hist["pred_low"] = close - band
            hist["pred_high"] = close + band
            hist = hist.dropna(subset=["pred_low", "pred_high"]).tail(120)
            chart = hist.set_index("date")[["actual_close", "pred_low", "pred_high"]]
            st.line_chart(chart)
            st.caption("Fallback band shows rolling volatility (20-day std of returns).")

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
    # ROW 3: Past Prediction table | Helpful Info | Actions (same layout)
    # -----------------------------
    r3_left, r3_mid, r3_right = st.columns([4, 4, 2])

    with r3_left:
        st.subheader("Past Prediction w/ News")

        if (not using_placeholders) and (feat_df is not None) and (model is not None):
            hist_table = build_prediction_history(feat_df, model, lookback=15)

            if not hist_table.empty:
                st.dataframe(hist_table, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough history to compute prediction errors yet.")
        else:
            st.info("Prediction history requires a trained model.")
    with r3_mid:
        st.subheader("Helpful Information")
        st.write(f"- **Model:** {p.get('model_name', 'LightGBM')}")
        st.write(f"- **Volatility (20d):** {p['vol20']:.4f}")
        st.write(f"- **Momentum (5d):** {p['mom5']:.4f}")
        st.write("- **Data Sources:** yfinance OHLCV + Finnhub headlines → sentiment aggregates")
        if "pred_ret" in p:
            st.write(f"- **Predicted next-day return:** {p['pred_ret']:+.4f}")
        if "pred_close" in p:
            st.write(f"- **Predicted close (center):** ${p['pred_close']:.2f}")

    with r3_right:
        st.subheader("Actions")
        st.button("Add to Watch list", use_container_width=True)
        st.button("Report Inaccuracy", use_container_width=True)
        with st.expander("How we get our prediction"):
            st.write(
                """
**High level flow:**
- Pull OHLCV → `raw_ohlcv.parquet`
- Pull Finnhub headlines → VADER sentiment → daily aggregates → `news_daily.parquet`
- Feature engineering (returns, lags, rolling stats, MA ratios + shifted news)
- Train LightGBM regression with time-series split
- Predict next-day return → convert to predicted close range + confidence proxy + risk
                """
            )