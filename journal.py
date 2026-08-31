"""Personal estimate journal (JSONL). Not a broker blotter."""

import json
from datetime import datetime, timezone

from research_policy import ROOT

JOURNAL_PATH = ROOT / "logs" / "estimates.jsonl"


def log_estimate(ticker: str, prediction: dict) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticker": ticker,
        "as_of": prediction.get("as_of"),
        "last_close": prediction.get("last_close"),
        "estimate_close": prediction.get("price"),
        "estimate_return": prediction.get("mean_return"),
        "naive_return": "0",
        "session": prediction.get("session_label"),
        "desk": prediction.get("desk") or {},
        "signal": {
            "lean": (prediction.get("signal") or {}).get("lean"),
            "action": (prediction.get("signal") or {}).get("action"),
            "stamp": (prediction.get("signal") or {}).get("stamp"),
            "execute": (prediction.get("signal") or {}).get("execute"),
        },
    }
    with JOURNAL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
