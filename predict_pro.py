import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from textblob import TextBlob
from datetime import datetime, time
from zoneinfo import ZoneInfo
from ticker_utils import CURRENCY_SYMBOLS

WATCHLIST = [
    ("RELIANCE.NS", "Reliance"),
    ("TCS.NS", "TCS"),
    ("HDFCBANK.NS", "HDFC Bank"),
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "Nvidia"),
    ("SHEL.L", "Shell"),
    ("7203.T", "Toyota"),
    ("0700.HK", "Tencent"),
    ("BHP.AX", "BHP"),
]


def _venue_clock(ticker):
    rules = [
        ((".NS", ".BO"), "Asia/Kolkata", time(9, 15), time(15, 30), "NSE"),
        ((".L", ".IL"), "Europe/London", time(8, 0), time(16, 30), "LSE"),
        ((".T",), "Asia/Tokyo", time(9, 0), time(15, 0), "TSE"),
        ((".HK",), "Asia/Hong_Kong", time(9, 30), time(16, 0), "HKEX"),
        ((".AX",), "Australia/Sydney", time(10, 0), time(16, 0), "ASX"),
        ((".TO", ".V"), "America/Toronto", time(9, 30), time(16, 0), "TSX"),
        ((".DE", ".F"), "Europe/Berlin", time(9, 0), time(17, 30), "Xetra"),
        ((".PA", ".AS"), "Europe/Paris", time(9, 0), time(17, 30), "Euronext"),
        ((".KS", ".KQ"), "Asia/Seoul", time(9, 0), time(15, 30), "KRX"),
        ((".TW",), "Asia/Taipei", time(9, 0), time(13, 30), "TWSE"),
        ((".SI",), "Asia/Singapore", time(9, 0), time(17, 0), "SGX"),
    ]
    tz_name, open_t, close_t, venue = "America/New_York", time(9, 30), time(16, 0), "US"
    for suffixes, zone, o, c, name in rules:
        if ticker.endswith(suffixes):
            tz_name, open_t, close_t, venue = zone, o, c, name
            break
    return tz_name, open_t, close_t, venue


def _market_session(ticker):
    tz_name, open_t, close_t, venue = _venue_clock(ticker)
    now = datetime.now(ZoneInfo(tz_name))
    if now.weekday() >= 5:
        return False, f"Weekend · {venue} closed"
    if open_t <= now.time() <= close_t:
        return True, f"{venue} session open"
    if now.time() < open_t:
        return False, f"Pre-open · {venue} closed"
    return False, f"After close · {venue} closed"


def format_market_cap(value, currency="USD"):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if n <= 0:
        return "N/A"
    symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    if currency == "INR":
        crore = n / 1e7
        if crore >= 1e5:
            return f"{symbol}{crore / 1e5:.2f} Lakh Cr"
        if crore >= 100:
            return f"{symbol}{crore:,.0f} Cr"
        if crore >= 1:
            return f"{symbol}{crore:,.1f} Cr"
        return f"{symbol}{n:,.0f}"
    if n >= 1e12:
        return f"{symbol}{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{symbol}{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{symbol}{n / 1e6:.2f}M"
    return f"{symbol}{n:,.0f}"


def get_news_items(ticker, limit=5):
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return items
    for article in raw[:12]:
        content = article.get("content") or {}
        title = content.get("title") or article.get("title") or ""
        if not title:
            continue
        provider = ""
        if isinstance(content.get("provider"), dict):
            provider = content["provider"].get("displayName") or ""
        provider = provider or article.get("publisher") or ""
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            node = content.get(key)
            if isinstance(node, dict):
                url = node.get("url") or ""
            if url:
                break
        url = url or article.get("link") or ""
        items.append({"title": title.strip(), "publisher": provider, "url": url})
        if len(items) >= limit:
            break
    return items


def get_sentiment_score(ticker, news_items=None):
    try:
        titles = [item["title"] for item in (news_items or get_news_items(ticker)) if item.get("title")]
        if not titles:
            return 0.0
        scores = [TextBlob(title).sentiment.polarity for title in titles]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("shortName") or ticker.replace(".NS", ""),
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "market_cap": info.get("marketCap") or 0,
            "country": info.get("country") or "",
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "currency": info.get("currency") or ("INR" if ticker.endswith(".NS") else "USD"),
            "last_trade": info.get("regularMarketPrice") or info.get("currentPrice"),
            "last_trade_time": info.get("regularMarketTime"),
            "previous_close": info.get("regularMarketPreviousClose"),
        }
    except Exception:
        return {
            "name": ticker.replace(".NS", ""),
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": 0,
            "country": "India",
            "pe": None,
            "forward_pe": None,
            "dividend_yield": None,
            "beta": None,
            "currency": "INR" if ticker.endswith(".NS") else "USD",
            "last_trade": None,
            "last_trade_time": None,
            "previous_close": None,
        }


def _sparkline(values, width=240, height=48, pad=2):
    closes = np.asarray(values, dtype=float)
    closes = closes[np.isfinite(closes)]
    if len(closes) < 2:
        return "", False
    mn, mx = float(closes.min()), float(closes.max())
    rng = mx - mn if mx > mn else 1.0
    n = len(closes)
    pts = []
    for i, v in enumerate(closes):
        x = pad + (i / (n - 1)) * (width - 2 * pad)
        y = pad + (1 - (v - mn) / rng) * (height - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts), bool(closes[-1] >= closes[0])


def _trade_local(rmt, tz_name):
    if rmt in (None, ""):
        return None
    try:
        if isinstance(rmt, datetime):
            dt = rmt
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            return dt.astimezone(ZoneInfo(tz_name))
        return datetime.fromtimestamp(int(rmt), tz=ZoneInfo("UTC")).astimezone(ZoneInfo(tz_name))
    except (TypeError, ValueError, OSError):
        return None


def _append_live_close(df, company_info, tz_name):
    """Yahoo daily bars often lag the last session. Patch Close from last trade when newer."""
    price = company_info.get("last_trade")
    trade_local = _trade_local(company_info.get("last_trade_time"), tz_name)
    if price is None:
        return df
    try:
        price = float(price)
        if not np.isfinite(price):
            return df
    except (TypeError, ValueError):
        return df
    bar = pd.Timestamp(df.index[-1])
    if bar.tzinfo is not None:
        bar = bar.tz_convert("UTC").tz_localize(None)
    bar_day = pd.Timestamp(bar.date())
    if trade_local is None:
        trade_day = bar_day
    else:
        trade_day = pd.Timestamp(trade_local.date())
    df = df.copy()
    if trade_day <= bar_day:
        df.iloc[-1, df.columns.get_loc("Close")] = price
        return df
    extra = {col: df[col].iloc[-1] if col in df.columns else np.nan for col in df.columns}
    extra["Close"] = price
    for col in ("Open", "High", "Low"):
        if col in extra:
            extra[col] = price
    if "Volume" in extra and (extra["Volume"] is None or (isinstance(extra["Volume"], float) and np.isnan(extra["Volume"]))):
        extra["Volume"] = 0.0
    row = pd.DataFrame([extra], index=[trade_day])
    return pd.concat([df, row])


def predict_pro(ticker="RELIANCE.NS", horizon=1):
    try:
        from tensorflow.keras.models import load_model
        model = load_model("pro_model.h5", compile=False)
        assets = joblib.load("pro_scaler.pkl")
        scaler = assets["scaler"]

        df = yf.download(ticker, period="1y", auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return f"Error: No price data for {ticker}."
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])
        if df.empty:
            return f"Error: No price data for {ticker}."

        nifty = yf.download("^NSEI", period="1y", progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty = nifty["Close"]

        usd_inr = yf.download("INR=X", period="1y", progress=False)
        if isinstance(usd_inr.columns, pd.MultiIndex):
            usd_inr.columns = usd_inr.columns.get_level_values(0)
        usd_inr = usd_inr["Close"]

        sp500 = yf.download("^GSPC", period="1y", progress=False)
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = sp500.columns.get_level_values(0)
        sp500 = sp500["Close"]

        nifty_vals = nifty.values
        sp_vals = sp500.values
        usd_vals = usd_inr.values
        if len(nifty_vals) < 21 or len(sp_vals) < 21 or len(usd_vals) < 21:
            return "Error: Not enough global marker data."
        if np.isnan(nifty_vals[-1]):
            nifty_vals = nifty_vals[:-1]
        if np.isnan(sp_vals[-1]):
            sp_vals = sp_vals[:-1]
        if np.isnan(usd_vals[-1]):
            usd_vals = usd_vals[:-1]
        if len(nifty_vals) < 21 or len(sp_vals) < 21 or len(usd_vals) < 21:
            return "Error: Not enough global marker data after NaN handling."

        nifty_trend = float((nifty_vals[-1] - nifty_vals[-20]) / nifty_vals[-20])
        sp_trend = float((sp_vals[-1] - sp_vals[-20]) / sp_vals[-20])
        usd_trend = float((usd_vals[-1] - usd_vals[-20]) / usd_vals[-20])
        bias = (nifty_trend * 0.5) + (sp_trend * 0.3) + (usd_trend * 0.2)
        bias = np.clip(bias, -0.02, 0.02)

        df["Nifty50"] = nifty
        df["USD_INR"] = usd_inr
        df["SP500"] = sp500

        news_items = get_news_items(ticker)
        sentiment_score = get_sentiment_score(ticker, news_items)
        company_info = get_company_info(ticker)
        tz_name, _, _, _ = _venue_clock(ticker)
        df = _append_live_close(df, company_info, tz_name)
        sentiment_bias = np.clip(sentiment_score * 0.5, -0.01, 0.01)
        total_bias = np.clip(bias + sentiment_bias, -0.03, 0.03)

        features_11 = [
            "Close", "Nifty50", "USD_INR", "SP500", "MA5", "MA20",
            "Return", "Volatility", "RSI", "RelVol", "Price_to_Vol",
        ]
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["Return"] = df["Close"].pct_change(fill_method=None)
        df["Volatility"] = df["Return"].rolling(window=5).std()
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        df["RelVol"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
        df["Price_to_Vol"] = df["Close"] / (df["Volume"] + 1)
        df[features_11] = df[features_11].ffill()
        df = df.dropna(subset=features_11)

        X = df[features_11].values
        if len(X) < 60:
            return f"Error: Not enough historical data for {ticker}."

        last_price_val = float(df["Close"].values[-1])
        prev_close = float(df["Close"].values[-2]) if len(df) > 1 else last_price_val
        last_window = X[-60:]
        last_scaled = scaler.transform(last_window)
        last_scaled = np.expand_dims(last_scaled, axis=0)
        predicted_return_ss = float(model.predict(last_scaled, verbose=0)[0][0])

        recent_volatility = float(df["Volatility"].values[-1]) if len(df["Volatility"].values) > 0 else 0.02
        uncertainty_factor = recent_volatility * 0.5
        ci_lower = np.clip(predicted_return_ss - 1.96 * uncertainty_factor, -0.05, 0.05)
        ci_upper = np.clip(predicted_return_ss + 1.96 * uncertainty_factor, -0.05, 0.05)

        horizons = [1, 5, 10, 20]
        horizon_predictions = {}
        horizon_cis = {}
        pred_return = predicted_return_ss
        for h in horizons:
            h_return = pred_return * h
            h_unc = uncertainty_factor * np.sqrt(h)
            h_ci_lower = np.clip(predicted_return_ss - 1.96 * h_unc, -0.05, 0.05)
            h_ci_upper = np.clip(predicted_return_ss + 1.96 * h_unc, -0.05, 0.05)
            horizon_predictions[h] = f"{h_return:.4%}"
            horizon_cis[h] = f"{h_ci_lower:.4%} - {h_ci_upper:.4%}"

        total_adjustment = predicted_return_ss + total_bias
        total_adjustment = np.clip(total_adjustment, predicted_return_ss - 0.03, predicted_return_ss + 0.03)
        final_price = last_price_val * (1 + total_adjustment)
        final_price_ci_lower = last_price_val * (1 + ci_lower + total_bias)
        final_price_ci_upper = last_price_val * (1 + ci_upper + total_bias)

        baseline_pred = predicted_return_ss
        importance_scores = {}
        last_window = X[-60:]
        for feat_idx, feat_name in enumerate(features_11):
            perturbations = []
            for _ in range(3):
                perturbed_window = last_window.copy()
                noise = np.random.normal(0, 0.1 * np.std(perturbed_window[:, feat_idx]))
                perturbed_window[:, feat_idx] += noise
                pert_input = np.expand_dims(perturbed_window, axis=0)
                try:
                    p = float(model.predict(pert_input, verbose=0)[0][0])
                    perturbations.append(abs(p - baseline_pred))
                except Exception:
                    perturbations.append(0)
            importance_scores[feat_name] = np.mean(perturbations) if perturbations else 0
        sorted_imp = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_imp[:3] if len(sorted_imp) >= 3 else sorted_imp
        top_trust_signal = top_features[0][0] if top_features else "N/A"

        recent_returns = df["Return"].values[-60:] if len(df["Return"].values) >= 60 else df["Return"].values
        annual_vol = float(np.std(recent_returns) * np.sqrt(252))
        var_95 = float(-np.percentile(recent_returns, 5)) * last_price_val
        daily_vol = float(np.std(recent_returns))
        stop_loss_price = last_price_val * (1 - 2 * daily_vol) if daily_vol > 0 else last_price_val
        take_profit_price = last_price_val * (1 + predicted_return_ss + abs(total_bias))

        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        rsi = float(df["RSI"].values[-1])
        ma5 = float(df["MA5"].values[-1])
        ma20 = float(df["MA20"].values[-1])
        rel_vol = float(df["RelVol"].values[-1])
        last_volume = float(df["Volume"].values[-1])
        high_52w = float(df["High"].max()) if "High" in df.columns else float(df["Close"].max())
        low_52w = float(df["Low"].min()) if "Low" in df.columns else float(df["Close"].min())
        day_change = last_price_val - prev_close
        day_change_pct = day_change / prev_close if prev_close else 0.0
        realized_5d = float(df["Close"].values[-1] / df["Close"].values[-6] - 1) if len(df) > 6 else 0.0
        realized_20d = float(df["Close"].values[-1] / df["Close"].values[-21] - 1) if len(df) > 21 else 0.0
        dist_from_high = last_price_val / high_52w - 1 if high_52w else 0.0

        nifty_series = df["Nifty50"].astype(float)
        stock_ret = df["Return"].astype(float)
        nifty_ret = nifty_series.pct_change()
        aligned = pd.concat([stock_ret, nifty_ret], axis=1).dropna()
        aligned.columns = ["s", "n"]
        if len(aligned) > 20 and float(aligned["n"].var()) > 0:
            beta_nifty = float(aligned["s"].cov(aligned["n"]) / aligned["n"].var())
            nifty_1d = float(aligned["n"].iloc[-1])
        else:
            beta_nifty = None
            nifty_1d = 0.0

        mean_r = float(np.mean(recent_returns))
        std_r = float(np.std(recent_returns))
        sharpe_60d = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0

        capital = 100000.0
        risk_budget = capital * 0.01
        stop_dist = last_price_val - stop_loss_price
        shares = int(risk_budget / stop_dist) if stop_dist > 1e-6 else 0
        position_notional = shares * last_price_val

        sparkline, spark_up = _sparkline(df["Close"].values[-90:])
        tz_name, open_t, close_t, venue = _venue_clock(ticker)
        session_open, session_label = _market_session(ticker)
        now_local = datetime.now(ZoneInfo(tz_name))
        last_trade_val = company_info.get("last_trade")
        last_trade_label = ""
        last_trade_at = ""
        try:
            if last_trade_val is not None:
                last_trade_val = float(last_trade_val)
                last_trade_label = f"{last_trade_val:.2f}"
        except (TypeError, ValueError):
            last_trade_val = None
            last_trade_label = ""
        trade_local = _trade_local(company_info.get("last_trade_time"), tz_name)
        if trade_local is not None:
            last_trade_at = trade_local.strftime("%d %b %Y %H:%M")
            as_of = trade_local.strftime("%d %b %Y")
            close_for_quote = last_trade_val if last_trade_val is not None else last_price_val
        else:
            bar_ts = pd.Timestamp(df.index[-1])
            if bar_ts.tzinfo is not None:
                as_of = bar_ts.tz_convert("UTC").strftime("%d %b %Y")
            else:
                as_of = bar_ts.strftime("%d %b %Y")
            close_for_quote = last_price_val
        if now_local.weekday() >= 5:
            quote_note = f"Weekend · last completed {venue} session is {as_of}."
        else:
            quote_note = f"Last close is the {venue} session for {as_of}."
        if last_trade_label and last_trade_at:
            quote_note += f" Last trade {last_trade_label} at {last_trade_at} {tz_name.split('/')[-1]} (Yahoo, often delayed)."
        elif last_trade_label:
            quote_note += f" Last trade {last_trade_label} (Yahoo, often delayed)."

        ma_trend = "Above MA20" if last_price_val >= ma20 else "Below MA20"
        ma_cross = "MA5 > MA20" if ma5 >= ma20 else "MA5 < MA20"

        pe = company_info.get("pe")
        pe_label = f"{pe:.1f}" if isinstance(pe, (int, float)) and np.isfinite(pe) else "N/A"
        dy = company_info.get("dividend_yield")
        if isinstance(dy, (int, float)) and np.isfinite(dy):
            dy_label = f"{dy:.2%}" if dy < 1 else f"{dy:.2f}%"
        else:
            dy_label = "N/A"

        warnings = []
        if daily_vol > 0.03:
            warnings.append("Realized volatility is elevated versus a typical large-cap day.")
        if abs(total_bias) > 0.015:
            warnings.append("Macro overlay (Nifty, S&P 500, USD/INR) is large relative to the stock signal.")
        if rsi >= 70:
            warnings.append("RSI is in overbought territory; mean-reversion risk is higher.")
        elif rsi <= 30:
            warnings.append("RSI is in oversold territory; bounce risk is two-sided.")
        if rel_vol < 0.6:
            warnings.append("Volume is thin versus the 20-day average; prints may be less reliable.")
        elif rel_vol > 2.0:
            warnings.append("Volume is more than 2× the 20-day average; a catalyst may already be in the price.")
        if dist_from_high < -0.2:
            warnings.append("Price is more than 20% below the 52-week high.")

        ccy = company_info.get("currency") or "USD"
        money = CURRENCY_SYMBOLS.get(ccy, f"{ccy} ")

        vs_nifty = day_change_pct - nifty_1d

        def pct(x):
            return f"{x:+.2%}"

        return {
            "price": f"{final_price:.2f}",
            "ci_lower": f"{final_price_ci_lower:.2f}",
            "ci_upper": f"{final_price_ci_upper:.2f}",
            "mean_return": f"{predicted_return_ss:.4%}",
            "ci_return_lower": f"{ci_lower:.4%}",
            "ci_return_upper": f"{ci_upper:.4%}",
            "bias": f"{total_bias:.4%}",
            "annual_volatility": f"{annual_vol:.2%}",
            "var_95": f"{money}{var_95:.2f}",
            "stop_loss": f"{money}{stop_loss_price:.2f}",
            "take_profit": f"{money}{take_profit_price:.2f}",
            "model_type": "Ensemble LSTM (price path + macro bias + news)",
            "sentiment": sentiment_label,
            "sentiment_color": "text-green-400" if sentiment_label == "Positive" else (
                "text-red-400" if sentiment_label == "Negative" else "text-slate-400"
            ),
            "company_name": company_info["name"],
            "company_sector": company_info["sector"],
            "company_industry": company_info["industry"],
            "market_cap": format_market_cap(company_info["market_cap"], ccy),
            "currency_symbol": money,
            "news_sentiment": f"{sentiment_score:.3f}",
            "feature_importance": [{"feature": name, "impact": f"{impact * 100:.2f}%"} for name, impact in top_features],
            "top_trust_signal": top_trust_signal,
            "horizon_1_return": horizon_predictions.get(1, "N/A"),
            "horizon_1_ci": horizon_cis.get(1, "N/A"),
            "horizon_5_return": horizon_predictions.get(5, "N/A"),
            "horizon_5_ci": horizon_cis.get(5, "N/A"),
            "horizon_10_return": horizon_predictions.get(10, "N/A"),
            "horizon_10_ci": horizon_cis.get(10, "N/A"),
            "horizon_20_return": horizon_predictions.get(20, "N/A"),
            "horizon_20_ci": horizon_cis.get(20, "N/A"),
            "volatility_warning": bool(daily_vol > 0.03),
            "bias_magnitude_warning": bool(abs(total_bias) > 0.015),
            "last_close": f"{close_for_quote:.2f}",
            "last_trade": last_trade_label,
            "last_trade_at": last_trade_at,
            "quote_note": quote_note,
            "prev_close": f"{prev_close:.2f}",
            "day_change": f"{day_change:+.2f}",
            "day_change_pct": pct(day_change_pct),
            "day_up": bool(day_change_pct >= 0),
            "predicted_move": f"{final_price - last_price_val:+.2f}",
            "as_of": as_of,
            "last_volume": f"{last_volume:,.0f}",
            "rel_vol": f"{rel_vol:.2f}×",
            "rsi": f"{rsi:.1f}",
            "ma_trend": ma_trend,
            "ma_cross": ma_cross,
            "high_52w": f"{high_52w:.2f}",
            "low_52w": f"{low_52w:.2f}",
            "dist_from_high": pct(dist_from_high),
            "realized_5d": pct(realized_5d),
            "realized_20d": pct(realized_20d),
            "beta_nifty": f"{beta_nifty:.2f}" if beta_nifty is not None else "N/A",
            "sharpe_60d": f"{sharpe_60d:.2f}",
            "pe": pe_label,
            "dividend_yield": dy_label,
            "nifty_20d": pct(nifty_trend),
            "spx_20d": pct(sp_trend),
            "usd_20d": pct(usd_trend),
            "nifty_1d": pct(nifty_1d),
            "vs_nifty": pct(vs_nifty),
            "sparkline": sparkline,
            "spark_up": bool(spark_up),
            "news": news_items,
            "warnings": warnings,
            "session_open": bool(session_open),
            "session_label": session_label,
            "shares": shares,
            "position_notional": f"{money}{position_notional:,.0f}",
            "risk_note": f"1% of {money}100,000 to the 2σ stop (illustrative, not a live order).",
            "disclaimer": (
                "Model estimate from delayed public data. Not a live price, not investment advice."
            ),
            "data_vendor": "Yahoo Finance",
            "estimate_kind": "next_session_close_from_last_close",
        }

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    result = predict_pro("RELIANCE.NS")
    if isinstance(result, dict):
        print(f"Professional Prediction for RELIANCE.NS (Ensemble Mode):")
        print(f"  Predicted Price: {result['price']}")
        print(f"  Last close: {result['last_close']} ({result['day_change_pct']})")
        print(f"  95% CI: {result['ci_lower']} - {result['ci_upper']}")
        print(f"  Mean Return: {result['mean_return']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Company: {result['company_name']}")
        print(f"  News: {len(result['news'])} items")
    else:
        print(f"Professional Prediction for RELIANCE.NS (Ensemble Mode): {result}")
