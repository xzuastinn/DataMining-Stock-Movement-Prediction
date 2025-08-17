# pip install lxml yfinance

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List
from config import *

import numpy as np
import pandas as pd
import yfinance as yf



# -----------------------------------------------------------------------------
# DATA COLLECTION HELPERS
# -----------------------------------------------------------------------------

def get_sp500_tickers() -> List[str]:
    """Return today's S&P 500 membership with symbols formatted for Yahoo.

    Yahoo Finance uses *hyphens* for class shares (e.g. ``BRK-B``) whereas the
    Wikipedia table lists them with dots (``BRK.B``).  Rather than hard-coding
    specific exceptions, we simply replace any dot with a hyphen so the script
    automatically adapts to future dot-class tickers without manual edits.
    """

    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    symbols = table["Symbol"].tolist()
    return [s.replace(".", "-") for s in symbols]


# -----------------------------------------------------------------------------
# PRICE HISTORY DOWNLOAD + TIDYING
# -----------------------------------------------------------------------------
ROLL = lambda s, w: s.rolling(window=w, min_periods=w)
EMA = lambda s, span: s.ewm(span=span, adjust=False).mean()


def multistock_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert *yfinance*'s wide multi-column frame into long format.

    After stacking level-0 (ticker) we get an index (Date, Ticker) and columns
    for OHLCV fields.  Works for single-ticker frames as well by normalising
    the column layout first.
    """

    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([["SINGLE"], raw.columns])

    long = (
        raw.stack(level=0, future_stack=True)  # → rows (Date, Ticker)
        .rename_axis(["Date", "Ticker"])
        .sort_index()
    )
    return long


def fetch_history(tickers: List[str], years: int = DEFAULT_LOOKBACK_YEARS) -> pd.DataFrame:
    """Download historical daily prices for *tickers* and return a tidy frame."""

    end = datetime.now(timezone.utc)
        
    start = end - timedelta(days=365 * years + 5)

    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i : i + BATCH_SIZE]
        try:
            raw = yf.download(
                " ".join(chunk),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] batch failed {chunk[:5]}… → {e}")
            time.sleep(REQUEST_PAUSE)
            continue

        frames.append(multistock_df(raw))
        time.sleep(REQUEST_PAUSE)

    if not frames:
        raise RuntimeError("No price data downloaded - check ticker list / network.")

    df = pd.concat(frames)

    # Ensure Adj Close exists (fallback to Close).
    if "Adj Close" not in df.columns:
        if "Close" not in df.columns:
            raise RuntimeError("Downloaded data lacks both Adj Close and Close.")
        df["Adj Close"] = df["Close"]

    return df[["Adj Close"]].dropna()


# -----------------------------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def add_simple_returns(df: pd.DataFrame, horizons=(1, 2, 3, 4, 5, 20, 60, 120)) -> pd.DataFrame:
    for h in horizons:
        df[f"Ret_{h}d"] = df.groupby("Ticker")["Adj Close"].pct_change(periods=h, fill_method=None)
    return df


def add_volatility(df: pd.DataFrame, windows=(10, 20, 60)) -> pd.DataFrame:
    for w in windows:
        df[f"Vol_{w}d"] = df.groupby("Ticker")["Adj Close"].transform(
            lambda x: ROLL(x.pct_change(fill_method=None), w).std()
        )
    return df


def add_macd(df: pd.DataFrame, short=12, long=26, signal=9) -> pd.DataFrame:
    def _macd(series: pd.Series) -> pd.DataFrame:
        macd_line = EMA(series, short) - EMA(series, long)
        signal_line = EMA(macd_line, signal)
        return pd.DataFrame({"MACD": macd_line, "MACD_Signal": signal_line})

    macd_df = df.groupby("Ticker")["Adj Close"].apply(_macd).reset_index(level=0, drop=True)
    return df.join(macd_df)


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    def _rsi(series: pd.Series) -> pd.Series:
        delta = series.diff()
        up = np.maximum(delta, 0.0)
        down = np.maximum(-delta, 0.0)
        roll_up = ROLL(pd.Series(up, index=series.index), window).mean()
        roll_down = ROLL(pd.Series(down, index=series.index), window).mean()
        rs = roll_up / roll_down
        return 100 - (100 / (1 + rs))

    df["RSI14"] = df.groupby("Ticker")["Adj Close"].transform(_rsi)
    return df


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    return (
        df_raw.copy()
        .pipe(add_simple_returns)
        .pipe(add_volatility)
        .pipe(add_macd)
        .pipe(add_rsi)
        .dropna(how="all")
    )


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] Retrieving S&P 500 ticker list...")
    tickers = get_sp500_tickers()

    print(f"[INFO] Downloading {DEFAULT_LOOKBACK_YEARS}y of price data for {len(tickers)} tickers...")
    df_raw = fetch_history(tickers)

    print("[INFO] Engineering features...")
    df_feat = engineer_features(df_raw)

    out_path = os.path.join(OUTPUT_DIR, "sp500_features.csv")
    df_feat.to_csv(out_path, encoding='utf-8')
    print(f"[DONE] Feature matrix saved -> {out_path} (rows: {len(df_feat):,})")


if __name__ == "__main__":
    main()
