"""Walk-forward next-day return metrics vs naive and persistence baselines.

Does not claim the LSTM beats the market. Writes model_metrics.json for /health and the UI.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research_policy import DATA_VENDOR, METRICS_PATH, MODEL_ID


def next_day_returns(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    closes = closes[np.isfinite(closes)]
    if len(closes) < 3:
        return np.array([], dtype=float)
    return (closes[1:] - closes[:-1]) / np.clip(closes[:-1], 1e-9, None)


def score_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    n = int(len(actual))
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "directional_accuracy": None}
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # Ignore near-zero actuals so a flat tape does not inflate accuracy.
    active = np.abs(actual) > 1e-8
    if int(active.sum()) == 0:
        da = None
    else:
        da = float(np.mean(np.sign(actual[active]) == np.sign(predicted[active])))
    return {"n": n, "mae": round(mae, 6), "rmse": round(rmse, 6), "directional_accuracy": da}


def walk_forward_baselines(closes: np.ndarray) -> dict:
    """Evaluate next-day return forecasts on a held-out tail (no look-ahead)."""
    rets = next_day_returns(closes)
    if len(rets) < 40:
        raise ValueError("Need at least 41 closes for a walk-forward split.")
    # Last 20% of return observations is the test window.
    split = int(len(rets) * 0.8)
    test = rets[split:]
    naive = np.zeros_like(test)
    persist = rets[split - 1 : split - 1 + len(test)]
    return {
        "naive_zero_return": score_forecast(test, naive),
        "persistence_last_return": score_forecast(test, persist),
        "test_start_index": split,
        "methodology": (
            "Time-ordered split (first 80% unused for these baseline scores, last 20% tested). "
            "Target is next-session simple return. Naive predicts 0. Persistence predicts "
            "the previous session’s return. No LSTM weights are used here."
        ),
    }


def merge_lstm(payload: dict, lstm_rows: dict) -> dict:
    from datetime import datetime, timezone

    tickers = payload.setdefault("tickers", {})
    for ticker, row in lstm_rows.items():
        bucket = tickers.setdefault(ticker, {})
        bucket["lstm_walk_forward"] = row
    payload["lstm_scored_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["note"] = (
        "Baselines plus LSTM out-of-sample MAE on the last 20% of sequences. "
        "If beats_naive_on_mae is false, a zero-change forecast was closer than the LSTM "
        "on that window — do not size a personal trade on the LSTM for that name."
    )
    return payload


def evaluate_tickers(closes_by_ticker: dict[str, np.ndarray]) -> dict:
    rows = {}
    for ticker, closes in closes_by_ticker.items():
        try:
            rows[ticker] = walk_forward_baselines(closes)
        except ValueError as exc:
            rows[ticker] = {"error": str(exc)}
    return {
        "model_id": MODEL_ID,
        "data_vendor": DATA_VENDOR,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Baselines only. If naive MAE is lower than persistence, a zero-change "
            "forecast beat last-return on that window. The LSTM is not scored in CI."
        ),
        "tickers": rows,
    }


def download_closes(tickers: list[str], period: str = "2y") -> dict[str, np.ndarray]:
    import yfinance as yf

    out = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            continue
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        series = df["Close"].dropna().astype(float).values
        if len(series) >= 41:
            out[ticker] = series
    return out


def main():
    parser = argparse.ArgumentParser(description="Write walk-forward baseline metrics.")
    parser.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "RELIANCE.NS"])
    parser.add_argument("--out", default=str(METRICS_PATH))
    args = parser.parse_args()
    payload = evaluate_tickers(download_closes(args.tickers))
    path = Path(args.out)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
