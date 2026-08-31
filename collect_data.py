import yfinance as yf
import pandas as pd
import sys

def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data found for {ticker}")
        return None
    
    filename = f"{ticker}_data.csv"
    data.to_csv(filename)
    print(f"Data saved to {filename}")
    return filename

if __name__ == "__main__":
    # Default to RELIANCE.NS (NSE) if no ticker provided
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    download_stock_data(ticker, start_date, end_date)
