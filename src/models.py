"""
models.py
---------
Model architectures for stock price forecasting:
    - build_rnn_model()   : a stacked SimpleRNN baseline
    - build_lstm_model()  : a stacked LSTM network (primary model)

Both take sequences of shape (window_size, n_features) and output a
single scalar: the predicted next-day (scaled) closing price.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_rnn_model(input_shape, units=50, dropout=0.2, learning_rate=0.001):
    """Baseline SimpleRNN model for comparison against LSTM."""
    model = Sequential([
        SimpleRNN(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        SimpleRNN(units, return_sequences=False),
        Dropout(dropout),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model


def build_lstm_model(input_shape, units=50, dropout=0.2, learning_rate=0.001):
    """Stacked LSTM model - the primary forecasting model for this project."""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model
