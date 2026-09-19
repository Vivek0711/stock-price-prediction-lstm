"""
preprocess.py
-------------
Turns a raw OHLCV dataframe into scaled sequences suitable for
training an RNN / LSTM time-series model.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data: np.ndarray, window_size: int = 60):
    """
    Convert a 1-D (or 2-D) array of scaled prices into supervised
    learning sequences: X = past `window_size` days, y = next day.

    Parameters
    ----------
    data        : np.ndarray, shape (n_samples, n_features)
    window_size : int, number of past time steps used to predict the next one

    Returns
    -------
    X : np.ndarray, shape (n_samples - window_size, window_size, n_features)
    y : np.ndarray, shape (n_samples - window_size,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])  # predict the first column (Close price)
    return np.array(X), np.array(y)


def prepare_data(df, feature_col: str = "Close", window_size: int = 60, train_split: float = 0.8):
    """
    Full preprocessing pipeline:
      1. Extract the target column (default: Close price)
      2. Scale to [0, 1] with MinMaxScaler
      3. Split chronologically into train / test (no shuffling - time series!)
      4. Build sliding-window sequences for supervised learning

    Returns
    -------
    X_train, y_train, X_test, y_test, scaler
    """
    # Some yfinance downloads have MultiIndex columns; flatten if needed
    if isinstance(df.columns, pd_multiindex_type()):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    prices = df[[feature_col]].values.astype("float64")

    split_idx = int(len(prices) * train_split)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices)  # fit ONLY on train data to avoid lookahead bias

    scaled_train = scaler.transform(train_prices)
    # include the last `window_size` train points so the test set has context
    scaled_test = scaler.transform(
        np.concatenate([train_prices[-window_size:], test_prices], axis=0)
    )

    X_train, y_train = create_sequences(scaled_train, window_size)
    X_test, y_test = create_sequences(scaled_test, window_size)

    return X_train, y_train, X_test, y_test, scaler


def pd_multiindex_type():
    import pandas as pd
    return pd.MultiIndex
