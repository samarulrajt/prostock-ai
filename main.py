import os
import json
import logging

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
from predict_pro import predict_pro as sync_predict, WATCHLIST
from ticker_utils import normalize_ticker
from nlp_brief import attach_brief, resolve_query, resolve_ticker_local
from research_policy import (
    DISCLAIMER,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    research_fields,
)
from journal import log_estimate
from trade_signal import attach_signal
from model_store import install_weights, model_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("prostock")

app = FastAPI(
    title="ProStock",
    description=DISCLAIMER,
    version="1.0.0",
)
templates = Jinja2Templates(directory="templates")


async def fetch_prediction_async(ticker: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sync_predict, ticker)


async def run_sync(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


def format_prediction_for_template(prediction_result, ticker=""):
    if isinstance(prediction_result, str):
        return {"type": "error", "value": prediction_result}
    payload = dict(prediction_result)
    payload["type"] = "success"
    payload.update(research_fields())
    attach_signal(payload, ticker)
    return payload


def page(request, **kwargs):
    ctx = {"watchlist": WATCHLIST, **research_fields(), "model_files": model_status()}
    ctx.update(kwargs)
    return templates.TemplateResponse(request, "index.html", ctx)


def _upload_allowed(request: Request, token: str = "") -> bool:
    expected = os.getenv("MODEL_UPLOAD_TOKEN", "").strip()
    provided = (token or "").strip() or (request.headers.get("X-Model-Token") or "").strip()
    if expected:
        return provided == expected
    host = (request.client.host if request.client else "") or ""
    return host in ("127.0.0.1", "::1", "testclient", "localhost")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return page(request)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, ticker: str = Form("")):
    raw = ticker.strip()
    try:
        symbol = await run_sync(resolve_query, raw)
        if not symbol:
            return page(
                request,
                ticker=raw,
                error="Could not map that to a ticker. Try AAPL or RELIANCE.NS.",
            )
        res = await fetch_prediction_async(symbol)
        formatted = format_prediction_for_template(res, symbol)
        if formatted["type"] == "error":
            return page(request, ticker=symbol, error=formatted["value"])
        log.info("predict ticker=%s as_of=%s", symbol, formatted.get("as_of"))
        try:
            log_estimate(symbol, formatted)
        except OSError:
            log.exception("estimate journal write failed")
        brief = await run_sync(attach_brief, raw, symbol, formatted)
        formatted["brief"] = brief
        return page(request, ticker=symbol, prediction=formatted)
    except Exception as e:
        return page(request, ticker=raw, error=str(e))


@app.get("/health")
async def health():
    model = MODEL_PATH.is_file()
    scaler = SCALER_PATH.is_file()
    metrics = METRICS_PATH.is_file()
    body = {
        "ok": model and scaler,
        "model_file": model,
        "scaler_file": scaler,
        "metrics_file": metrics,
        **research_fields(),
    }
    return JSONResponse(body, status_code=200 if body["ok"] else 503)


@app.get("/api/model")
async def api_model_status():
    return {"ok": True, **model_status()}


@app.post("/api/model")
async def api_model_upload(
    request: Request,
    model: UploadFile = File(...),
    scaler: UploadFile = File(...),
    metrics: UploadFile | None = File(None),
    token: str = Form(""),
):
    if not _upload_allowed(request, token):
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)
    model_bytes = await model.read()
    scaler_bytes = await scaler.read()
    metrics_bytes = await metrics.read() if metrics and metrics.filename else None
    try:
        status = install_weights(model_bytes, scaler_bytes, metrics_bytes)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    log.info("model weights replaced model=%s scaler=%s", model.filename, scaler.filename)
    return {"ok": True, "files": status}


@app.post("/upload-model", response_class=HTMLResponse)
async def upload_model_form(
    request: Request,
    model: UploadFile = File(...),
    scaler: UploadFile = File(...),
    metrics: UploadFile | None = File(None),
    token: str = Form(""),
):
    if not _upload_allowed(request, token):
        return page(request, error="Model upload not authorized. Set MODEL_UPLOAD_TOKEN or use localhost.")
    model_bytes = await model.read()
    scaler_bytes = await scaler.read()
    metrics_bytes = await metrics.read() if metrics and metrics.filename else None
    try:
        install_weights(model_bytes, scaler_bytes, metrics_bytes)
    except ValueError as exc:
        return page(request, error=str(exc))
    return page(request, error=None, uploaded="Weights replaced. Analyze a ticker to use the new model.")


@app.get("/api/meta")
async def api_meta():
    metrics = None
    if METRICS_PATH.is_file():
        try:
            metrics = json.loads(METRICS_PATH.read_text())
        except json.JSONDecodeError:
            metrics = {"error": "model_metrics.json is not valid JSON"}
    return {"ok": True, **research_fields(), "metrics": metrics}


@app.get("/api/resolve")
async def api_resolve(q: str = ""):
    query = (q or "").strip()
    if not query:
        return JSONResponse({"ok": False, "error": "q is required"}, status_code=400)
    symbol = resolve_ticker_local(query)
    if not symbol:
        return JSONResponse({"ok": False, "error": "Could not resolve a ticker", "query": query}, status_code=400)
    return {"ok": True, "query": query, "ticker": symbol}


@app.get("/api/predict/{ticker}")
async def api_predict(ticker: str):
    ticker = normalize_ticker(ticker)
    res = await fetch_prediction_async(ticker)
    if isinstance(res, str):
        return JSONResponse({"ok": False, "error": res, "ticker": ticker}, status_code=400)
    log.info("api_predict ticker=%s", ticker)
    payload = dict(res)
    payload.update(research_fields())
    attach_signal(payload, ticker)
    try:
        log_estimate(ticker, payload)
    except OSError:
        log.exception("estimate journal write failed")
    return {"ok": True, "ticker": ticker, "prediction": payload}


@app.post("/api/ask")
async def api_ask(payload: dict):
    """Natural-language entry: extract ticker, run the model, return a research note."""
    query = str(payload.get("query") or payload.get("ticker") or "").strip()
    if not query:
        return JSONResponse({"ok": False, "error": "query is required"}, status_code=400)
    symbol = await run_sync(resolve_query, query)
    if not symbol:
        return JSONResponse({"ok": False, "error": "Could not resolve a ticker"}, status_code=400)
    res = await fetch_prediction_async(symbol)
    if isinstance(res, str):
        return JSONResponse({"ok": False, "error": res, "ticker": symbol}, status_code=400)
    payload = dict(res)
    payload.update(research_fields())
    attach_signal(payload, symbol)
    brief = await run_sync(attach_brief, query, symbol, payload)
    try:
        log_estimate(symbol, payload)
    except OSError:
        log.exception("estimate journal write failed")
    return {
        "ok": True,
        "query": query,
        "ticker": symbol,
        "brief": brief,
        "prediction": payload,
        **research_fields(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
