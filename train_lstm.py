import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import sys

def get_sentiment_score(ticker):
    # In a real scenario, we'd use a News API (like NewsAPI.org)
    # For this implementation, we simulate sentiment based on basic price trends
    # to show how the feature would be integrated into the LSTM model.
    # Mock sentiment: +1 for upward trend, -1 for downward
    try:
        data = yf.download(ticker, period="5d", progress=False)
        if len(data) < 2: return 0
        trend = data['Close'].iloc[-1] - data['Close'].iloc[0]
        return 1.0 if trend > 0 else -1.0
    except:
        return 0

def preprocess_for_lstm(df, window_size=60):
    # Feature Engineering
    df = df.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=5).std()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume analysis: Relative Volume (Current Vol / Avg Vol)
    df['RelVol'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # Add simulated sentiment (in a real app, this would be pre-calculated per day)
    # We simulate a daily sentiment score based on price action for training
    df['Sentiment'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)
    
    df = df.dropna()
    
    features = ['Close', 'MA5', 'MA20', 'Return', 'Volatility', 'RSI', 'RelVol', 'Sentiment']
    X = df[features].values
    y = df['Close'].shift(-1).fillna(df['Close']).values
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for LSTM
    X_seq, y_seq = [], []
    for i in range(window_size, len(X_scaled)):
        X_seq.append(X_scaled[i-window_size:i])
        y_seq.append(y[i])
        
    return np.array(X_seq), np.array(y_seq), scaler

def train_lstm(ticker="RELIANCE.NS"):
    print(f"Training LSTM model for {ticker}...")
    df = yf.download(ticker, period="2y", progress=False)
    
    X, y, scaler = preprocess_for_lstm(df)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Model Architecture
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    
    # Save
    model.save('stock_lstm_model.h5')
    joblib.dump(scaler, 'lstm_scaler.pkl')
    print("LSTM Model and Scaler saved!")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    train_lstm(ticker)
