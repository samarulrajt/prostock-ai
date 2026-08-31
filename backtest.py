import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
TICKER = "RELIANCE.NS"
WINDOW_SIZE = 60
N_FOLDS = 3
TEST_SIZE = 30

# Load model and scaler
print("Loading model and scaler...")
try:
    model = load_model('pro_model.h5', compile=False)
    assets = joblib.load('pro_scaler.pkl')
    scaler = assets['scaler']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Fetch data
print(f"Fetching data for {TICKER}...")
df = yf.download(TICKER, period="2y", progress=False)
if df.empty:
    print("No data fetched.")
    exit()

# Feature engineering (11 features matching model)
features = ['Close', 'MA5', 'MA20', 'Return', 'Volatility', 'RSI', 'RelVol', 'Price_to_Vol',
            'Nifty50', 'USD_INR', 'SP500']

df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['Return'] = df['Close'].pct_change(fill_method=None)
df['Volatility'] = df['Return'].rolling(window=5).std()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + (gain/loss)))

df['RelVol'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
df['Price_to_Vol'] = df['Close'] / (df['Volume'] + 1)

# Fill global placeholders
for col in ['Nifty50', 'USD_INR', 'SP500']:
    df[col] = 0.0

df = df.dropna()
X = df[features].values

# Scale all data at once
X_scaled = scaler.transform(X)

# =======================
# WALK-FORWARD VALIDATION
# =======================
print(f"\n--- Walk-Forward Cross-Validation ({N_FOLDS} folds) ---")

all_preds = []
all_actual = []
fold_accuracies = []

fold_start = WINDOW_SIZE

for fold in range(1, N_FOLDS + 1):
    fold_test_end = fold_start + TEST_SIZE
    if fold_test_end + WINDOW_SIZE > len(df):
        print(f"Fold {fold}: Not enough data, stopping.")
        break
    
    fold_train_end = fold_test_end - WINDOW_SIZE
    train_data = df.iloc[:fold_train_end]
    test_data = df.iloc[fold_train_end:fold_test_end]
    
    # Scale
    scaler_fold = joblib.load('pro_scaler.pkl')  # Re-load to be safe
    scaler_fold = scaler_fold['scaler']
    X_train_scaled = scaler_fold.transform(train_data[features].values)
    X_test_scaled = scaler_fold.transform(test_data[features].values)
    
    # Create sequences from test data
    X_seq, y_seq = [], []
    for i in range(WINDOW_SIZE, len(X_test_scaled) + WINDOW_SIZE):
        # Index within test_scaled
        tidx = i - WINDOW_SIZE
        if 0 <= tidx < len(X_test_scaled) - WINDOW_SIZE + 1:
            X_seq.append(X_test_scaled[tidx:tidx + WINDOW_SIZE])
            # Actual next-day return from original close prices
            actual_idx = len(train_data) + tidx
            if actual_idx + 1 < len(df['Close'].values):
                actual_ret = df['Close'].pct_change().shift(-1).fillna(0).values[actual_idx + 1]
                y_seq.append(actual_ret)
    
    if len(X_seq) == 0 or len(y_seq) == 0:
        print(f"Fold {fold}: No sequences generated, skipping.")
        fold_start += TEST_SIZE
        continue
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Predict
    preds = model.predict(X_seq, verbose=0).flatten()
    
    # Evaluate
    acc = (np.sign(y_seq) == np.sign(preds)).mean()
    mae = mean_absolute_error(y_seq, preds)
    
    all_preds.extend(preds.tolist())
    all_actual.extend(y_seq.tolist())
    fold_accuracies.append(acc)
    
    print(f"Fold {fold}: MAE={mae*100:.2f}%, Accuracy={acc*100:.2f}% ({len(y_seq)} samples)")
    
    # Move window forward
    fold_start += TEST_SIZE

# =======================
# FALLBACK: SIMPLE TRAIN/TEST
# =======================
if len(fold_accuracies) == 0:
    print("\n--- Simple Train/Test Backtest ---")
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(WINDOW_SIZE, len(X_scaled)):
        X_seq.append(X_scaled[i-WINDOW_SIZE:i])
        if i + 1 < len(df['Close'].values):
            y_seq.append(df['Close'].pct_change().shift(-1).fillna(0).values[i + 1])
    
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    split2 = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split2], X_seq[split2:]
    y_train, y_test = y_seq[:split2], y_seq[split2:]
    
    print("Predicting on test set...")
    preds = model.predict(X_test, verbose=0).flatten()
    
    acc = (np.sign(y_test) == np.sign(preds)).mean()
    mae = mean_absolute_error(y_test, preds)
    
    print(f"Test samples: {len(y_test)}")
    print(f"MAE: {mae*100:.2f}%, Directional Accuracy: {acc*100:.2f}%")
    print(f"Naive MAE (no change): {mean_absolute_error(y_test, np.zeros_like(y_test))*100:.2f}%")
    
    # Plot
    last_price = df['Close'].iloc[0]
    actual_curve = [last_price]
    pred_curve = [last_price]
    for r in y_test:
        actual_curve.append(actual_curve[-1] * (1 + r))
    for r in preds:
        pred_curve.append(pred_curve[-1] * (1 + r))
    
    dates = df.index[WINDOW_SIZE:WINDOW_SIZE+len(actual_curve)-1]
    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual_curve, label='Actual', color='blue')
    plt.plot(dates, pred_curve, label='Predicted', color='orange', linestyle='--')
    plt.title(f"Backtest: {TICKER}")
    plt.xlabel("Date"); plt.ylabel("Price (Rs)"); plt.legend()
    plt.savefig("backtest_reliance.png")
    print(f"Plot saved as backtest_reliance.png")
    
    print(f"\nAccuracy: {acc*100:.2f}% - {'Good' if acc > 0.55 else 'Needs improvement'}")
else:
    # Walk-forward summary
    avg_acc = np.mean(fold_accuracies) * 100
    print(f"\n--- Walk-Forward Results ---")
    print(f"Successful folds: {len(fold_accuracies)}/{N_FOLDS}")
    print(f"Average Directional Accuracy: {avg_acc:.2f}%")
    print(f"(Higher than 55% = meaningful predictive power)")
    
    # Plot combined results
    if len(all_preds) > 0 and len(all_actual) > 0:
        plt.figure(figsize=(10, 4))
        # Show first fold's price movement
        n = min(60, len(all_actual))
        actual_ret = all_actual[:n]
        pred_ret = all_preds[:n]
        
        price = df['Close'].iloc[WINDOW_SIZE]
        actual_price = [price]
        pred_price = [price]
        for r in actual_ret:
            actual_price.append(actual_price[-1] * (1 + r))
        for r in pred_ret:
            pred_price.append(pred_price[-1] * (1 + r))
        
        dates = df.index[WINDOW_SIZE:WINDOW_SIZE+n]
        plt.plot(dates, actual_price, label='Actual', color='blue')
        plt.plot(dates, pred_price, label='Walk-Forward Pred', color='orange', linestyle='--')
        plt.title(f"Walk-Forward: {TICKER}")
        plt.xlabel("Date"); plt.ylabel("Price (Rs)"); plt.legend()
        plt.savefig("backtest_reliance_walkforward.png")
        print(f"Walk-forward plot saved as backtest_reliance_walkforward.png")
