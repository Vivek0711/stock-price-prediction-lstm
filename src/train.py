"""
train.py
--------
End-to-end pipeline:
    1. Download historical stock data (Yahoo Finance)
    2. Preprocess into scaled sliding-window sequences
    3. Train a SimpleRNN model (baseline) and an LSTM model (main model)
    4. Evaluate both on the held-out test set (RMSE / MAE / MAPE)
    5. Save trained models and comparison plots to disk

Usage
-----
    python src/train.py --ticker AAPL --start 2015-01-01 --end 2024-12-31
"""

import argparse
import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import download_stock_data
from preprocess import prepare_data
from models import build_rnn_model, build_lstm_model
from utils import evaluate, plot_predictions, plot_training_history

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNN & LSTM models for stock price prediction")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=60, help="Lookback window size (days)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data
    df = download_stock_data(args.ticker, args.start, args.end)
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        df, feature_col="Close", window_size=args.window
    )
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    results = {}

    # 2. Baseline SimpleRNN
    print("\n=== Training SimpleRNN baseline ===")
    rnn_model = build_rnn_model(input_shape)
    rnn_history = rnn_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    rnn_model.save(os.path.join(MODEL_DIR, f"{args.ticker}_rnn_model.keras"))
    plot_training_history(
        rnn_history, "SimpleRNN Training History",
        os.path.join(OUTPUT_DIR, f"{args.ticker}_rnn_loss.png"),
    )

    # 3. LSTM (main model)
    print("\n=== Training LSTM model ===")
    lstm_model = build_lstm_model(input_shape)
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    lstm_model.save(os.path.join(MODEL_DIR, f"{args.ticker}_lstm_model.keras"))
    plot_training_history(
        lstm_history, "LSTM Training History",
        os.path.join(OUTPUT_DIR, f"{args.ticker}_lstm_loss.png"),
    )

    # 4. Evaluate both models on the test set
    for name, model in [("RNN", rnn_model), ("LSTM", lstm_model)]:
        pred_scaled = model.predict(X_test)
        pred = scaler.inverse_transform(pred_scaled)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        metrics = evaluate(actual, pred)
        results[name] = metrics
        print(f"\n{name} test metrics: {metrics}")

        dates = df.index[-len(actual):]
        plot_predictions(
            dates, actual.flatten(), pred.flatten(),
            f"{args.ticker} - {name}: Actual vs Predicted Close Price",
            os.path.join(OUTPUT_DIR, f"{args.ticker}_{name.lower()}_predictions.png"),
        )

    # 5. Summary
    print("\n================ Summary ================")
    for name, metrics in results.items():
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print("===========================================")
    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Plots saved to:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
