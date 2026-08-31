"""Out-of-sample LSTM next-day return scores vs the same split as eval_baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yfinance as yf

from eval_baselines import download_closes, evaluate_tickers, merge_lstm, score_forecast
from research_policy import METRICS_PATH, MODEL_PATH, SCALER_PATH
from train_pro_model import FEATURES, ProStockEngine, flatten_ohlcv


def _lstm_oos(ticker: str, engine: ProStockEngine, model, scaler, nifty, usd, sp) -> dict:
    raw = yf.download(ticker, period="2y", progress=False, auto_adjust=False, threads=False)
    df = flatten_ohlcv(raw)
    if df is None or df.empty or "Close" not in df.columns:
        return {"error": "no history"}
    feat = engine.engineer_features(df, nifty, usd, sp)
    if len(feat) < engine.window_size + 25:
        return {"error": "not enough rows after features"}
    table = feat[FEATURES].astype(np.float32)
    closes = table["Close"].values
    y = np.zeros(len(closes), dtype=np.float32)
    y[:-1] = (closes[1:] - closes[:-1]) / np.clip(closes[:-1], 1e-6, None)
    X_scaled = scaler.transform(table.values).astype(np.float32)
    w = engine.window_size
    last_i = len(X_scaled) - 1
    indices = list(range(w, last_i))
    if len(indices) < 40:
        return {"error": "not enough sequences"}
    split = int(len(indices) * 0.8)
    test_idx = indices[split:]
    preds, actuals = [], []
    for i in test_idx:
        window = np.expand_dims(X_scaled[i - w : i], axis=0)
        hat = float(model.predict(window, verbose=0)[0][0])
        preds.append(hat)
        actuals.append(float(y[i]))
    test_hat = np.asarray(preds)
    test_y = np.asarray(actuals)
    naive = np.zeros_like(test_y)
    scored = score_forecast(test_y, test_hat)
    naive_scored = score_forecast(test_y, naive)
    mae_m, mae_n = scored.get("mae"), naive_scored.get("mae")
    beats_naive = mae_m is not None and mae_n is not None and mae_m < mae_n
    return {
        "lstm": scored,
        "naive_zero_return_same_window": naive_scored,
        "beats_naive_on_mae": beats_naive,
        "methodology": (
            "Same 80/20 time split on sequences: window of 60 feature rows predicts "
            "next-session simple return. No shuffling."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Score saved LSTM out-of-sample and update metrics.")
    parser.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "RELIANCE.NS"])
    parser.add_argument("--out", default=str(METRICS_PATH))
    args = parser.parse_args()
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise SystemExit("Need pro_model.h5 and pro_scaler.pkl")

    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH, compile=False)
    assets = joblib.load(SCALER_PATH)
    scaler = assets["scaler"]
    engine = ProStockEngine()
    nifty, usd, sp = engine.download_macros()
    lstm_rows = {}
    for ticker in args.tickers:
        print(f"LSTM OOS {ticker}...")
        lstm_rows[ticker] = _lstm_oos(ticker, engine, model, scaler, nifty, usd, sp)

    path = Path(args.out)
    if path.is_file():
        payload = json.loads(path.read_text())
    else:
        payload = evaluate_tickers(download_closes(args.tickers))
    payload = merge_lstm(payload, lstm_rows)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
