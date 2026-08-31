import json
import os
import re
from urllib.parse import quote

import requests

from ticker_utils import normalize_ticker

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nemotron-3-ultra:cloud")
# API keys authenticate https://ollama.com, not local :11434 (that needs `ollama signin`).
if ":cloud" in OLLAMA_MODEL and ("11434" in OLLAMA_URL or OLLAMA_URL.endswith("ollama")):
    OLLAMA_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com").rstrip("/")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").rstrip("/")
HTTP_TIMEOUT = int(os.getenv("NLP_TIMEOUT", "180"))
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

EXTRACT_SYSTEM = (
    "You map a user request to a Yahoo Finance ticker. "
    "Reply with JSON only: {\"ticker\": \"SYMBOL\"}. "
    "Use exchange suffixes when needed (.NS India, .L London, .T Tokyo, .HK Hong Kong, .AX Australia). "
    "US names have no suffix. If unknown, return {\"ticker\": \"\"}."
)

BRIEF_SYSTEM = (
    "You are a markets desk editor. Write a concise research note from the JSON facts. "
    "Two short paragraphs, no bullet lists. "
    "Open with the desk action and stamp (BUY/SELL/HOLD/NO TRADE). "
    "Do not invent figures that are not in the JSON. "
    "State that fills happen at the user's broker on delayed Yahoo data."
)

NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "shell": "SHEL.L",
    "toyota": "7203.T",
    "tencent": "0700.HK",
    "bhp": "BHP.AX",
}


def looks_like_ticker(raw: str) -> bool:
    text = (raw or "").strip()
    if not text or " " in text or "?" in text:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9.^.=-]{1,24}", text))


def slim_prediction(ticker: str, prediction: dict) -> dict:
    news = prediction.get("news") or []
    headlines = [item.get("title") for item in news[:3] if item.get("title")]
    return {
        "ticker": ticker,
        "company": prediction.get("company_name"),
        "as_of": prediction.get("as_of"),
        "last_close": prediction.get("last_close"),
        "predicted_close": prediction.get("price"),
        "expected_return": prediction.get("mean_return"),
        "sentiment": prediction.get("sentiment"),
        "rsi": prediction.get("rsi"),
        "trend": prediction.get("ma_trend"),
        "volatility": prediction.get("annual_volatility"),
        "pe": prediction.get("pe"),
        "flags": prediction.get("warnings") or [],
        "headlines": headlines,
        "session": prediction.get("session_label"),
        "desk_action": (prediction.get("signal") or {}).get("action"),
        "model_lean": (prediction.get("signal") or {}).get("lean"),
        "stamp": (prediction.get("signal") or {}).get("stamp"),
        "instruction": (prediction.get("signal") or {}).get("instruction"),
    }


def resolve_ticker_local(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in NAME_TO_TICKER:
        return NAME_TO_TICKER[lowered]
    for name, ticker in sorted(NAME_TO_TICKER.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return ticker
    if looks_like_ticker(text):
        return normalize_ticker(text)
    for token in re.findall(r"[A-Za-z0-9.^.=-]{2,24}", text):
        if token.lower() in NAME_TO_TICKER:
            return NAME_TO_TICKER[token.lower()]
        if looks_like_ticker(token) and (token.isupper() or "." in token or any(ch.isdigit() for ch in token)):
            return normalize_ticker(token)
    return ""


def write_brief_facts(ticker: str, prediction: dict) -> str:
    facts = slim_prediction(ticker, prediction)
    company = facts.get("company") or ticker
    return (
        f"{company} ({ticker}) last closed at {facts.get('last_close')} as of {facts.get('as_of')}. "
        f"The model estimates {facts.get('predicted_close')} ({facts.get('expected_return')}), "
        f"{facts.get('sentiment') or 'unspecified'} news tone, RSI {facts.get('rsi')}, "
        f"trend {facts.get('trend')}. "
        f"Desk action: {facts.get('desk_action') or 'n/a'} "
        f"({facts.get('stamp') or 'n/a'}; model lean {facts.get('model_lean') or 'n/a'}). "
        f"{facts.get('instruction') or 'This is a model estimate from delayed data.'}"
    )


def _ollama_chat(system: str, user: str) -> str:
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        headers=headers,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return ((payload.get("message") or {}).get("content") or "").strip()


def extract_ticker(query: str) -> str:
    local = resolve_ticker_local(query)
    if local:
        return local
    text = _ollama_chat(EXTRACT_SYSTEM, query)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return ""
    data = json.loads(match.group(0))
    return normalize_ticker(str(data.get("ticker") or ""))


def write_brief_ollama(ticker: str, prediction: dict) -> str:
    user = json.dumps(slim_prediction(ticker, prediction), ensure_ascii=False)
    return _ollama_chat(BRIEF_SYSTEM, user)


def write_brief_n8n(query: str, ticker: str, prediction: dict) -> dict:
    response = requests.post(
        N8N_WEBHOOK_URL,
        json={
            "query": query,
            "ticker": ticker,
            "prediction": slim_prediction(ticker, prediction) if prediction else None,
        },
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def resolve_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return ""
    local = resolve_ticker_local(query)
    if local:
        return local
    try:
        return extract_ticker(query)
    except Exception:
        return ""


def attach_brief(query: str, ticker: str, prediction: dict) -> str:
    if not prediction or prediction.get("type") == "error":
        return ""
    if N8N_WEBHOOK_URL:
        try:
            data = write_brief_n8n(query, ticker, prediction)
            brief = data.get("brief") or data.get("text") or ""
            if brief:
                return brief
        except Exception:
            pass
    try:
        return write_brief_ollama(ticker, prediction)
    except Exception:
        return write_brief_facts(ticker, prediction)


def predict_url(ticker: str, base: str) -> str:
    return f"{base.rstrip('/')}/api/predict/{quote(ticker)}"
