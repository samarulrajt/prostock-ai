import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import sys

def predict_next_day(ticker):
    try:
        # 1. Load model and scaler
        model = joblib.load('stock_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # 2. Get latest data
        print(f"Fetching latest data for {ticker}...")
        data = yf.download(ticker, period="1mo")
        if data.empty:
            return "No data found for this ticker."

        # 3. Preprocess latest data to get features for the last day
        df = data.copy()
        
        # Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Daily Return
        df['Return'] = df['Close'].pct_change()
        
        # Volatility
        df['Volatility'] = df['Return'].rolling(window=5).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Get the most recent row
        latest_features = df[['MA5', 'MA20', 'Return', 'Volatility', 'RSI', 'Close']].iloc[-1:]
        
        # Scale features
        latest_scaled = scaler.transform(latest_features)
        
        # Predict
        prediction = model.predict(latest_scaled)
        
        return f"Predicted next closing price for {ticker}: {prediction[0]:.2f}"
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    print(predict_next_day(ticker))
