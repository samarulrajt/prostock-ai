#!/usr/bin/env python3
"""Pull Yahoo-compatible tickers from NSE, US listings, and major world indices."""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from ticker_utils import normalize_ticker

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NSE_EQUITY = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
SP500_CSV = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

FALLBACK_WORLD = [
    "7203.T", "6758.T", "6861.T", "9984.T", "8306.T",
    "0700.HK", "0941.HK", "0005.HK", "1299.HK", "3690.HK",
    "005930.KS", "000660.KS", "035420.KS",
    "2330.TW", "2317.TW",
    "SAP.DE", "ASML.AS", "NESN.SW",
]

WIKI_INDICES = [
    ("https://en.wikipedia.org/wiki/FTSE_100_Index", ".L"),
    ("https://en.wikipedia.org/wiki/DAX", ".DE"),
    ("https://en.wikipedia.org/wiki/CAC_40", ".PA"),
    ("https://en.wikipedia.org/wiki/Hang_Seng_Index", ".HK"),
    ("https://en.wikipedia.org/wiki/S%26P/ASX_200", ".AX"),
    ("https://en.wikipedia.org/wiki/S%26P/TSX_60", ".TO"),
    ("https://en.wikipedia.org/wiki/Nikkei_225", ".T"),
]


def _get(url, timeout=45):
    response = requests.get(url, headers=UA, timeout=timeout)
    response.raise_for_status()
    return response


def unique_keep_order(tickers):
    seen = set()
    out = []
    for raw in tickers:
        t = normalize_ticker(str(raw))
        if not t or t in seen or t.startswith("#"):
            continue
        if any(ch in t for ch in ("^", "/", " ")):
            continue
        seen.add(t)
        out.append(t)
    return out


def get_nse_tickers():
    print("Fetching NSE equity list...")
    text = _get(NSE_EQUITY).text
    df = pd.read_csv(StringIO(text))
    col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
    tickers = [f"{str(s).strip()}.NS" for s in df[col].dropna()]
    print(f"  NSE: {len(tickers)}")
    return tickers


def _us_from_pipe(url, symbol_col, test_col="Test Issue"):
    text = _get(url).text
    df = pd.read_csv(StringIO(text), sep="|")
    if df.empty:
        return []
    if test_col in df.columns:
        df = df[df[test_col].astype(str).str.upper() != "Y"]
    if "ETF" in df.columns:
        # Keep ETFs; they are valid Yahoo symbols. No filter.
        pass
    symbols = df[symbol_col].dropna().astype(str)
    symbols = symbols[~symbols.str.contains(r"[\$]|File Creation", regex=True)]
    return symbols.tolist()


def get_us_tickers():
    print("Fetching US listings (NASDAQ / NYSE / ARCA)...")
    nasdaq = _us_from_pipe(NASDAQ_LISTED, "Symbol")
    other = _us_from_pipe(OTHER_LISTED, "ACT Symbol")
    tickers = unique_keep_order(nasdaq + other)
    print(f"  US: {len(tickers)}")
    return tickers


def get_sp500_tickers():
    print("Fetching S&P 500...")
    try:
        df = pd.read_csv(StringIO(_get(SP500_CSV).text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = unique_keep_order(df[col].dropna().astype(str))
        print(f"  S&P 500: {len(tickers)}")
        return tickers
    except Exception as e:
        print(f"  S&P 500 skipped: {e}")
        return []


def _symbol_column(df):
    for name in df.columns:
        n = str(name).strip().lower()
        if n in {"ticker", "symbol", "epic", "tidm", "code", "yahoo ticker"}:
            return name
        if "ticker" in n or n == "sym":
            return name
    return None


def get_wiki_index_tickers(url, suffix):
    html = _get(url).text
    tables = pd.read_html(StringIO(html))
    found = []
    for df in tables:
        col = _symbol_column(df)
        if col is None:
            continue
        for raw in df[col].dropna().astype(str):
            token = raw.split()[0].split(":")[-1].strip()
            token = token.replace(".", "-") if suffix == "" else token
            if not token or len(token) > 16:
                continue
            if suffix and not token.upper().endswith(suffix):
                # HK often stored as 0700 or 00700
                if suffix == ".HK" and token.replace(".", "").isdigit():
                    token = token.replace(".", "").zfill(4)
                found.append(f"{token}{suffix}")
            else:
                found.append(token)
        if len(found) >= 30:
            break
    return unique_keep_order(found)


def get_world_index_tickers():
    print("Fetching major non-US index constituents (Wikipedia)...")
    all_t = []
    for url, suffix in WIKI_INDICES:
        try:
            batch = get_wiki_index_tickers(url, suffix)
            print(f"  {url.split('/')[-1]}: {len(batch)}")
            all_t.extend(batch)
        except Exception as e:
            print(f"  skip {url}: {e}")
    return unique_keep_order(all_t)


def write_list(path, tickers):
    Path(path).write_text("\n".join(tickers) + ("\n" if tickers else ""))


def main():
    nse = []
    us = []
    sp500 = []
    world = []
    try:
        nse = get_nse_tickers()
    except Exception as e:
        print(f"NSE failed: {e}")
    try:
        us = get_us_tickers()
    except Exception as e:
        print(f"US listings failed: {e}")
    try:
        sp500 = get_sp500_tickers()
    except Exception as e:
        print(f"S&P 500 failed: {e}")
    try:
        world = get_world_index_tickers()
    except Exception as e:
        print(f"World indices failed: {e}")

    all_tickers = unique_keep_order(nse + us + sp500 + world + FALLBACK_WORLD)
    majors = unique_keep_order(sp500 + world + FALLBACK_WORLD)

    write_list("all_tickers.txt", all_tickers)
    write_list("major_tickers.txt", majors)
    print(f"Wrote {len(all_tickers)} symbols to all_tickers.txt")
    print(f"Wrote {len(majors)} symbols to major_tickers.txt (S&P 500 + world indices)")
    print("Train with:  python3 train_pro_model.py --file all_tickers.txt")
    print("Smaller job: python3 train_pro_model.py --file major_tickers.txt")


if __name__ == "__main__":
    main()
