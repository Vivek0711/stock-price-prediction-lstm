"""
utils.py
--------
Evaluation metrics and plotting helpers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(y_true, y_pred):
    """Return RMSE, MAE and MAPE for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


def plot_predictions(dates, y_true, y_pred, title, out_path):
    """Save a line chart comparing actual vs predicted prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual Price", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted Price", linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_training_history(history, title, out_path):
    """Save training vs validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")
