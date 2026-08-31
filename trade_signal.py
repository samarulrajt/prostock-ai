"""Next-session BUY / SELL / HOLD from the LSTM estimate.

Personal desk only. Execute at your broker. Delayed Yahoo is not a fill.
"""

from research_policy import desk_verdict

# Round-trip friction floor (~25 bps). Predicted move inside this is HOLD.
MIN_MOVE = 0.0025
# Predicted |return| must also clear half of OOS MAE or it is noise.
MAE_FRAC = 0.5


def parse_return(value):
    if value is None:
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text.upper() == "N/A":
        return None
    try:
        pct = float(text)
    except ValueError:
        return None
    return pct / 100.0 if abs(pct) > 1.0 or str(value).strip().endswith("%") else pct


def _deadband(mae):
    floor = MIN_MOVE
    if mae is None:
        return floor
    try:
        mae_f = float(mae)
    except (TypeError, ValueError):
        return floor
    return max(floor, abs(mae_f) * MAE_FRAC)


def _lean(predicted_return, band):
    if predicted_return is None:
        return "HOLD"
    if predicted_return > band:
        return "BUY"
    if predicted_return < -band:
        return "SELL"
    return "HOLD"


def build_signal(prediction: dict, ticker: str = "") -> dict:
    desk = prediction.get("desk") or desk_verdict(ticker)
    r = parse_return(prediction.get("mean_return"))
    mae = desk.get("lstm_mae")
    band = _deadband(mae)
    lean = _lean(r, band)
    scored = bool(desk.get("lstm_scored"))
    beats = desk.get("beats_naive") is True
    tradable = scored and beats

    if not scored:
        stamp = "DO NOT SIZE"
        tone = "blocked"
        action = "NO TRADE"
        instruction = (
            f"Model lean is {lean}. Do not place an order until this ticker is scored "
            f"with: python eval_lstm.py --tickers {ticker or 'SYMBOL'}"
        )
        reason = "LSTM not scored versus naive on this name."
    elif not beats:
        stamp = "DO NOT SIZE"
        tone = "blocked"
        action = "NO TRADE"
        instruction = (
            f"Model lean is {lean}. Stay flat — the LSTM lost to a zero-change forecast "
            "on the walk-forward window, so a buy/sell here is worse than doing nothing."
        )
        reason = "LSTM MAE did not beat naive."
    elif lean == "HOLD":
        stamp = "STAY FLAT"
        tone = "hold"
        action = "HOLD"
        instruction = (
            "Hold / no new trade. The estimated move is inside the error band "
            "(max of 0.25% and 0.5× LSTM MAE), so direction is noise."
        )
        reason = "Predicted return inside deadband."
    elif lean == "BUY":
        stamp = "VALIDATED"
        tone = "buy"
        action = "BUY"
        instruction = (
            "Buy for the next session (personal long). Use last close as the research "
            "entry, set the stop on the card, and size only what you can lose. Yahoo is delayed."
        )
        reason = "Estimated next-session return is above the deadband and LSTM beat naive."
    else:
        stamp = "VALIDATED"
        tone = "sell"
        action = "SELL"
        instruction = (
            "Sell / do not hold a long into the next session. This desk is long-only: "
            "go to cash rather than short unless you already have a short process."
        )
        reason = "Estimated next-session return is below the deadband and LSTM beat naive."

    execute = action in ("BUY", "SELL")
    return {
        "lean": lean,
        "action": action,
        "stamp": stamp,
        "tone": tone,
        "execute": execute,
        "tradable": tradable,
        "instruction": instruction,
        "reason": reason,
        "predicted_return": r,
        "deadband": round(band, 6),
        "horizon": "next_session",
        "rule": (
            "BUY if estimate > deadband, SELL if estimate < −deadband, else HOLD. "
            "Execute only when LSTM OOS MAE beats naive. Deadband = max(0.25%, 0.5×MAE)."
        ),
    }


def attach_signal(payload: dict, ticker: str) -> dict:
    payload["desk"] = payload.get("desk") or desk_verdict(ticker)
    payload["signal"] = build_signal(payload, ticker)
    return payload
