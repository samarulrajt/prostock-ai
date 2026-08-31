import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys

def preprocess_data(df):
    # Feature Engineering
    df = df.copy()
    
    # Use 'Close' price for predictions
    df['Target'] = df['Close'].shift(-1) # Predict next day's price
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Daily Return
    df['Return'] = df['Close'].pct_change()
    
    # Volatility (Rolling Std Dev)
    df['Volatility'] = df['Return'].rolling(window=5).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaNs created by rolling windows and shift
    df = df.dropna()
    
    return df

def train_model(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    # Skip the first two rows (Price and Ticker)
    df = df.iloc[2:].reset_index(drop=True)
    df.index.name = 'Date'
    df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Convert columns to numeric
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    df = preprocess_data(df)
    
    # Define features (X) and target (y)
    # We exclude target and original OHLC data to avoid leakage and focus on engineered features
    features = ['MA5', 'MA20', 'Return', 'Volatility', 'RSI', 'Close']
    X = df[features]
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Eval
    preds = model.predict(X_test_scaled)
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    print(f"R2 Score: {r2_score(y_test, preds):.2f}")
    
    # Save model and scaler
    joblib.dump(model, 'stock_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and Scaler saved!")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS_data.csv"
    train_model(csv_file)
