import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model
import sys

def get_sentiment_score(ticker):
    # Simulate sentiment: based on 5-day trend
    try:
        data = yf.download(ticker, period="5d", progress=False)
        if len(data) < 2: return 0
        trend = data['Close'].iloc[-1] - data['Close'].iloc[0]
        return 1.0 if trend > 0 else -1.0
    except:
        return 0

def predict_lstm(ticker="RELIANCE.NS"):
    try:
        # Load
        model = load_model('stock_lstm_model.h5')
        scaler = joblib.load('lstm_scaler.pkl')
        
        # Fetch data (need at least 60 days for window)
        df = yf.download(ticker, period="3mo", progress=False)
        
        # Feature Engineering
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Return'].rolling(window=5).std()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RelVol'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Current sentiment
        sentiment = get_sentiment_score(ticker)
        df['Sentiment'] = sentiment # Use latest sentiment for the whole window for simplicity
        
        df = df.dropna()
        
        features = ['Close', 'MA5', 'MA20', 'Return', 'Volatility', 'RSI', 'RelVol', 'Sentiment']
        X = df[features].values
        
        # Scale and sequence
        X_scaled = scaler.transform(X)
        last_window = X_scaled[-60:]
        last_window = np.expand_dims(last_window, axis=0)
        
        prediction = model.predict(last_window, verbose=0)
        return f"{prediction[0][0]:.2f}"
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    print(f"LSTM Prediction for {ticker}: {predict_lstm(ticker)}")
