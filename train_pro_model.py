import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from ticker_utils import normalize_ticker
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model

FEATURES = [
    "Close", "Nifty50", "USD_INR", "SP500",
    "MA5", "MA20", "Return", "Volatility", "RSI", "RelVol", "Price_to_Vol",
]


def flatten_ohlcv(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns=lambda c: str(c).title() if str(c).lower() == "adj close" else c)
    return out


def extract_ticker_frame(raw, ticker):
    if raw is None or raw.empty:
        return None
    if not isinstance(raw.columns, pd.MultiIndex):
        return flatten_ohlcv(raw)
    for level in range(raw.columns.nlevels):
        names = set(map(str, raw.columns.get_level_values(level)))
        if ticker in names:
            try:
                piece = raw.xs(ticker, axis=1, level=level)
            except ValueError:
                piece = raw[ticker]
            return flatten_ohlcv(piece)
    return None


def close_series(df):
    df = flatten_ohlcv(df)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return df["Close"].squeeze()


def load_tickers(path, limit=None):
    tickers = []
    for line in Path(path).read_text().splitlines():
        t = normalize_ticker(line)
        if not t or t.startswith("#"):
            continue
        tickers.append(t)
    seen = []
    for t in tickers:
        if t not in seen:
            seen.append(t)
    if limit:
        seen = seen[:limit]
    return seen


class ProStockEngine:
    def __init__(self, window_size=60, max_per_ticker=300):
        self.window_size = window_size
        self.max_per_ticker = max_per_ticker
        self.scaler = StandardScaler()

    def engineer_features(self, df, nifty_df, usd_df, sp_df):
        df = flatten_ohlcv(df)
        df = df.copy()
        df["Nifty50"] = nifty_df
        df["USD_INR"] = usd_df
        df["SP500"] = sp_df
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
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def sequences_from_scaled(self, X_scaled, y):
        X_seq, y_seq = [], []
        for i in range(self.window_size, len(X_scaled)):
            X_seq.append(X_scaled[i - self.window_size : i])
            y_seq.append(y[i])
        if not X_seq:
            return None, None
        X_seq = np.asarray(X_seq, dtype=np.float32)
        y_seq = np.asarray(y_seq, dtype=np.float32)
        if self.max_per_ticker and len(X_seq) > self.max_per_ticker:
            X_seq = X_seq[-self.max_per_ticker :]
            y_seq = y_seq[-self.max_per_ticker :]
        return X_seq, y_seq

    def download_macros(self):
        print("Downloading Nifty, USD/INR, S&P 500...")
        nifty = close_series(yf.download("^NSEI", period="2y", progress=False, auto_adjust=False))
        usd = close_series(yf.download("INR=X", period="2y", progress=False, auto_adjust=False))
        sp = close_series(yf.download("^GSPC", period="2y", progress=False, auto_adjust=False))
        if len(nifty) < 80 or len(usd) < 80 or len(sp) < 80:
            raise RuntimeError("Not enough macro history to train.")
        return nifty, usd, sp

    def download_batch(self, batch):
        if len(batch) == 1:
            raw = yf.download(batch[0], period="2y", progress=False, auto_adjust=False, threads=False)
            return {batch[0]: flatten_ohlcv(raw)}
        raw = yf.download(
            batch,
            period="2y",
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        frames = {}
        for ticker in batch:
            frames[ticker] = extract_ticker_frame(raw, ticker)
        return frames

    def collect_feature_tables(self, tickers, nifty, usd, sp, download_batch_size, pause_s):
        tables = {}
        skipped = []
        n = len(tickers)
        for start in range(0, n, download_batch_size):
            batch = tickers[start : start + download_batch_size]
            print(f"Download {start + 1}–{min(start + len(batch), n)} / {n}")
            try:
                frames = self.download_batch(batch)
            except Exception as e:
                print(f"  Batch failed ({e}); retrying one-by-one")
                frames = {}
                for ticker in batch:
                    try:
                        frames[ticker] = flatten_ohlcv(
                            yf.download(ticker, period="2y", progress=False, auto_adjust=False, threads=False)
                        )
                    except Exception as inner:
                        print(f"  Skip {ticker}: {inner}")
                        skipped.append(ticker)
                        frames[ticker] = None
            for ticker, df in frames.items():
                if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
                    skipped.append(ticker)
                    continue
                if df["Close"].dropna().shape[0] < self.window_size + 20:
                    skipped.append(ticker)
                    continue
                try:
                    eng = self.engineer_features(df, nifty, usd, sp)
                except Exception as e:
                    print(f"  Skip {ticker}: {e}")
                    skipped.append(ticker)
                    continue
                if len(eng) < self.window_size + 5:
                    skipped.append(ticker)
                    continue
                tables[ticker] = eng[FEATURES].astype(np.float32)
            if start + download_batch_size < n:
                time.sleep(pause_s)
        return tables, skipped

    def train_on_multiple(self, tickers, epochs=15, batch_size=64, download_batch_size=40, pause_s=1.0):
        print(f"Training universe: {len(tickers)} tickers")
        nifty, usd, sp = self.download_macros()
        tables, skipped = self.collect_feature_tables(
            tickers, nifty, usd, sp, download_batch_size, pause_s
        )
        if not tables:
            print("No usable ticker histories. Nothing to train.")
            return

        print(f"Fitting shared scaler on {len(tables)} stocks...")
        for feat in tables.values():
            self.scaler.partial_fit(feat.values)

        all_X, all_y, used = [], [], []
        for ticker, feat in tables.items():
            closes = feat["Close"].values
            y = np.zeros(len(closes), dtype=np.float32)
            y[:-1] = (closes[1:] - closes[:-1]) / np.clip(closes[:-1], 1e-6, None)
            X_scaled = self.scaler.transform(feat.values).astype(np.float32)
            X_seq, y_seq = self.sequences_from_scaled(X_scaled, y)
            if X_seq is None:
                continue
            all_X.append(X_seq)
            all_y.append(y_seq)
            used.append(ticker)

        X_final = np.concatenate(all_X)
        y_final = np.concatenate(all_y)
        print(f"Sequence tensor: {X_final.shape} from {len(used)} tickers ({len(skipped)} skipped)")

        rng = np.random.default_rng(42)
        order = rng.permutation(len(X_final))
        X_final = X_final[order]
        y_final = y_final[order]

        inputs = Input(shape=(self.window_size, X_final.shape[2]))
        path1 = LSTM(32, return_sequences=False)(inputs)
        path2 = LSTM(64, return_sequences=True)(inputs)
        path2 = LSTM(32, return_sequences=False)(path2)
        merged = Concatenate()([path1, path2])
        dense1 = Dense(32, activation="relu")(merged)
        dropout = Dropout(0.2)(dense1)
        output = Dense(1)(dropout)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer="adam", loss="mse")

        print("Training universal return predictor...")
        model.fit(
            X_final,
            y_final,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        )

        model.save("pro_model.h5")
        joblib.dump(
            {
                "scaler": self.scaler,
                "tickers_trained": used,
                "n_sequences": int(len(y_final)),
                "features": FEATURES,
            },
            "pro_scaler.pkl",
        )
        Path("train_universe.log").write_text(
            f"used={len(used)}\nskipped={len(skipped)}\nsequences={len(y_final)}\n"
            + "\n".join(used)
        )
        print("Universal return model saved to pro_model.h5")


def parse_args():
    p = argparse.ArgumentParser(description="Train the ProStock LSTM on NSE tickers.")
    p.add_argument("tickers", nargs="*", help="Optional explicit tickers (default: all_tickers.txt)")
    p.add_argument("--file", default="all_tickers.txt", help="Ticker list, one symbol per line")
    p.add_argument("--limit", type=int, default=None, help="Use only the first N tickers")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--download-batch", type=int, default=40)
    p.add_argument("--max-per-ticker", type=int, default=300)
    p.add_argument("--pause", type=float, default=1.0, help="Seconds between Yahoo batches")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tickers:
        tickers = [normalize_ticker(t) for t in args.tickers]
    else:
        tickers = load_tickers(args.file, limit=args.limit)
    engine = ProStockEngine(max_per_ticker=args.max_per_ticker)
    engine.train_on_multiple(
        tickers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        download_batch_size=args.download_batch,
        pause_s=args.pause,
    )
