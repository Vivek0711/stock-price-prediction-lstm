# 📈 Stock Price Prediction using RNN & LSTM

A time-series forecasting project that predicts stock closing prices from
historical market data using two deep learning architectures — a **SimpleRNN**
baseline and a **stacked LSTM** network — built with TensorFlow/Keras.

---

## 🚀 Features

- Downloads real historical OHLCV data for any ticker via `yfinance` (Yahoo Finance)
- Cleans and scales data with `MinMaxScaler`, using a chronological (no-shuffle) train/test split to avoid lookahead bias
- Builds sliding-window sequences (default: 60-day lookback) for supervised learning
- Trains and compares two models:
  - **SimpleRNN** — baseline recurrent model
  - **LSTM** — stacked long short-term memory network (primary model)
- Evaluates with RMSE, MAE, and MAPE
- Saves training-loss curves and actual-vs-predicted price charts
- Supports iterative multi-day future forecasting

## 🗂️ Project Structure

```
stock-price-prediction-lstm/
├── src/
│   ├── data_loader.py     # Download / cache historical stock data
│   ├── preprocess.py      # Scaling + sliding-window sequence generation
│   ├── models.py          # SimpleRNN and LSTM architectures
│   ├── train.py           # Main training + evaluation pipeline
│   ├── predict.py         # Forecast future prices with a saved model
│   └── utils.py           # Metrics + plotting helpers
├── data/                  # Cached CSVs of downloaded stock data
├── saved_models/          # Trained .keras model files
├── outputs/                # Generated plots (loss curves, predictions)
├── requirements.txt
└── README.md
```

## 🛠️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

### Train both models on a stock ticker

```bash
python src/train.py --ticker AAPL --start 2015-01-01 --end 2024-12-31 --epochs 25
```

This will:
1. Download AAPL historical data
2. Train a SimpleRNN and an LSTM model
3. Print RMSE / MAE / MAPE for both on the test set
4. Save trained models to `saved_models/` and plots to `outputs/`

**Optional arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | AAPL | Stock ticker symbol |
| `--start` | 2015-01-01 | Historical data start date |
| `--end` | 2024-12-31 | Historical data end date |
| `--window` | 60 | Lookback window (days) |
| `--epochs` | 25 | Training epochs |
| `--batch_size` | 32 | Training batch size |

### Forecast future prices with a trained model

```bash
python src/predict.py --ticker AAPL --model saved_models/AAPL_lstm_model.keras --days 10
```

Prints predicted closing prices for the next N trading days.

## 📊 Example Output

After training, check the `outputs/` folder for:
- `<TICKER>_lstm_loss.png` / `<TICKER>_rnn_loss.png` — training vs validation loss
- `<TICKER>_lstm_predictions.png` / `<TICKER>_rnn_predictions.png` — actual vs predicted closing price on the test set

## 🧠 How It Works

1. **Data collection** — pulls daily OHLCV data for the requested ticker and date range.
2. **Preprocessing** — the closing price series is scaled to `[0, 1]` and split chronologically (80/20) into train/test sets, since shuffling would leak future information into training.
3. **Sequence generation** — each training example is a 60-day window of past prices used to predict the next day's price.
4. **Modeling** — a stacked LSTM (3 layers + dropout) learns long-range temporal dependencies better than a plain SimpleRNN, which tends to struggle with vanishing gradients over long sequences.
5. **Evaluation** — predictions are inverse-scaled back to real price values and compared against actual prices using RMSE, MAE, and MAPE.

## ⚠️ Disclaimer

This project is for educational purposes only. Stock price prediction from
historical prices alone is inherently limited — markets are influenced by
countless external factors (news, macroeconomics, sentiment, etc.) not
captured in this model. **Do not use this for real financial decisions.**

## 📄 License

MIT License — feel free to use and adapt this project.
