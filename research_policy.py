"""Shared research-tool policy. This product is not a broker or advisor."""

import json
from pathlib import Path

MODEL_ID = "prostock-lstm-v1"
DATA_VENDOR = "Yahoo Finance"
DATA_VENDOR_NOTE = (
    "Quotes and history come from Yahoo Finance: unofficial, often delayed, "
    "and daily bars can lag the last trade. Not an exchange feed."
)
ESTIMATE_KIND = "next_session_close_from_last_close"
DISCLAIMER = (
    "Personal desk: each Analyze returns a next-session BUY, SELL, or HOLD. "
    "You place every order. Not investment advice. Delayed Yahoo is not a live fill. "
    "Execute only when the stamp is VALIDATED (LSTM MAE beats a naive zero-change forecast)."
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "pro_model.h5"
SCALER_PATH = ROOT / "pro_scaler.pkl"
METRICS_PATH = ROOT / "model_metrics.json"


def _metrics_row(payload: dict, ticker: str) -> dict:
    tickers = payload.get("tickers") or {}
    key = (ticker or "").strip()
    if not key:
        return {}
    candidates = [key, key.upper(), key.lower()]
    upper = key.upper()
    if upper.endswith(".NS"):
        candidates.append(upper[:-3])
    else:
        candidates.append(f"{upper}.NS")
    for cand in candidates:
        if cand in tickers:
            return tickers[cand]
        for stored, row in tickers.items():
            if stored.upper() == cand.upper():
                return row
    return {}


def desk_verdict(ticker: str) -> dict:
    """Whether the saved LSTM beat a naive zero-change forecast OOS for this ticker."""
    empty = {
        "lstm_scored": False,
        "beats_naive": None,
        "lstm_mae": None,
        "naive_mae": None,
        "hit_rate": None,
        "detail": "Run: python eval_lstm.py",
    }
    if not ticker or not METRICS_PATH.is_file():
        return empty
    try:
        payload = json.loads(METRICS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    row = _metrics_row(payload, ticker)
    lstm = row.get("lstm_walk_forward") or {}
    if lstm.get("error"):
        return {**empty, "detail": str(lstm["error"])}
    if not lstm:
        return empty
    scored = lstm.get("lstm") or {}
    return {
        "lstm_scored": True,
        "beats_naive": lstm.get("beats_naive_on_mae"),
        "lstm_mae": scored.get("mae"),
        "naive_mae": (lstm.get("naive_zero_return_same_window") or {}).get("mae"),
        "hit_rate": scored.get("directional_accuracy"),
        "detail": lstm.get("methodology") or "",
    }


def research_fields() -> dict:
    return {
        "disclaimer": DISCLAIMER,
        "data_vendor": DATA_VENDOR,
        "data_vendor_note": DATA_VENDOR_NOTE,
        "model_id": MODEL_ID,
        "estimate_kind": ESTIMATE_KIND,
    }
