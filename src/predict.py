"""
predict.py
----------
Load a trained LSTM (or RNN) model and forecast the next N days of
closing prices beyond the end of the historical dataset.

Usage
-----
    python src/predict.py --ticker AAPL --model saved_models/AAPL_lstm_model.keras --days 10
"""

import argparse
import os

import numpy as np
from tensorflow.keras.models import load_model

from data_loader import download_stock_data
from preprocess import prepare_data

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast future stock prices with a trained model")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--model", type=str, required=True, help="Path to a saved .keras model")
    parser.add_argument("--days", type=int, default=10, help="Number of future days to forecast")
    return parser.parse_args()


def forecast_future(model, last_window, scaler, n_days):
    """
    Iteratively predict `n_days` ahead, feeding each prediction back in
    as input for the next step (a common approach for multi-step
    time-series forecasting with a single-step model).
    """
    window = last_window.copy()
    predictions = []

    for _ in range(n_days):
        pred_scaled = model.predict(window.reshape(1, *window.shape), verbose=0)
        predictions.append(pred_scaled[0, 0])
        # slide the window forward by one step
        window = np.append(window[1:], [[pred_scaled[0, 0]]], axis=0)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)


def main():
    args = parse_args()

    df = download_stock_data(args.ticker, args.start, args.end)
    _, _, X_test, _, scaler = prepare_data(df, feature_col="Close", window_size=args.window)

    model = load_model(args.model)
    last_window = X_test[-1]  # most recent `window` days of scaled data

    future_prices = forecast_future(model, last_window, scaler, args.days)

    print(f"\nForecast for {args.ticker} - next {args.days} trading days:")
    for i, price in enumerate(future_prices.flatten(), start=1):
        print(f"  Day +{i}: ${price:.2f}")


if __name__ == "__main__":
    main()
