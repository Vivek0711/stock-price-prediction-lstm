"""
data_loader.py
---------------
Handles fetching historical stock market data.

Primary source: Yahoo Finance via the `yfinance` package.
Fallback: a local CSV file (data/<ticker>.csv) with columns
          Date, Open, High, Low, Close, Adj Close, Volume
          in case there is no internet access when running this project
          (useful for offline demos / grading environments).
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_stock_data(ticker: str, start: str, end: str, save: bool = True) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker between two dates.

    Parameters
    ----------
    ticker : str   e.g. "AAPL", "MSFT", "TSLA"
    start  : str   "YYYY-MM-DD"
    end    : str   "YYYY-MM-DD"
    save   : bool  if True, caches the data to data/<ticker>.csv

    Returns
    -------
    pd.DataFrame indexed by Date with columns:
        Open, High, Low, Close, Adj Close, Volume
    """
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    try:
        import yfinance as yf

        print(f"Downloading {ticker} data from Yahoo Finance ({start} -> {end}) ...")
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            raise ValueError("yfinance returned an empty dataframe")

        if save:
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(csv_path)
            print(f"Saved a local copy to {csv_path}")

        return df

    except Exception as e:
        print(f"[WARNING] Could not download live data ({e}).")
        if os.path.exists(csv_path):
            print(f"Falling back to cached file: {csv_path}")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df
        else:
            raise RuntimeError(
                f"No internet access and no cached file found at {csv_path}. "
                f"Place a CSV with Date/Open/High/Low/Close/Volume columns there, "
                f"or run this with an active internet connection."
            )


def load_local_csv(path: str) -> pd.DataFrame:
    """Load a stock CSV file that already exists on disk."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df
