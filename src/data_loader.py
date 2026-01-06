# This module acts as an API Collector to fetch data from Yahoo Finance.

import pandas as pd
import os

from typing import Optional

def fetch_crypto_data(coin_id: str, days: int) -> Optional[str]:
    """
    Fetches data and extracts RAW VALUES to prevent CSV corruption.
    """
    ticker_map = {
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "solana": "SOL-USD",
        "dogecoin": "DOGE-USD",
        "cardano": "ADA-USD"
    }
    
    ticker = ticker_map.get(coin_id.lower())
    if not ticker:
        print(f"[ERROR] Coin '{coin_id}' not supported.")
        return None
    print(f"[INFO] Checking cache for {ticker}...")
    os.makedirs("data/raw", exist_ok=True)
    filename = f"data/raw/{coin_id}_prices.csv"
    
    # Check if file exists
    # We implement local caching to avoid hitting the API rate limits and to speed up development.
    # Re-downloading static historical data is inefficient.
    if os.path.exists(filename):
        print(f"[INFO] Data found in cache: {filename}")
        return filename

    print(f"[INFO] Downloading data for {ticker}...")
    
    try:
        import yfinance as yf # Lazy import
        # We import yfinance here instead of at the top because it's a heavy library.
        # This prevents the entire app from crashing on startup if yfinance isn't installed
        # or if there's an internet issue, unless we explicitly call this function.
        
        # Fetch data using 'download' which is often more stable for formatting
        data = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        
        # 1. Flatten Headers (Handle MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            # Keep only the top level name (Open, High, Low, Close...)
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        
        # 2. Rename columns to lowercase for safety
        data.columns = [c.lower() for c in data.columns]
        
        # 3. Extract specific columns safely
        # We explicitly check if 'close' is a DataFrame (bad) or Series (good)
        price_series = data['close']
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0] # Take the first column
            
        vol_series = data['volume']
        if isinstance(vol_series, pd.DataFrame):
            vol_series = vol_series.iloc[:, 0]

        # 4. Construct Clean DataFrame
        df = pd.DataFrame()
        df['timestamp'] = data.index
        df['price'] = price_series.values # .values forces raw numpy array (No formatting issues)
        df['market_cap'] = 0.0
        df['total_volume'] = vol_series.values
        
        # 5. Filter days
        if len(df) > days:
            df = df.iloc[-days:]
            
        # 6. Save
        os.makedirs("data/raw", exist_ok=True)
        filename = f"data/raw/{coin_id}_prices.csv"
        df.to_csv(filename, index=False)
        
        # DEBUG: Print first row to prove it's numbers
        print(f"[DEBUG] First row saved: {df.iloc[0]['price']} (Should be a number)")
        print(f"[SUCCESS] Saved {len(df)} rows to {filename}")
        return filename
        
    except Exception as e:
        print(f"[WARN] yfinance failed: {e}. Trying direct HTTP download...")
        return download_via_requests(ticker, days, coin_id)

def download_via_requests(ticker: str, days: int, coin_id: str) -> Optional[str]:
    """
    Fallback method to download CSV directly from Yahoo Finance query API.
    Bypasses yfinance library issues on older Python versions.
    """
    import requests
    import time
    import io

    # Calculate timestamps
    end_time = int(time.time())
    start_time = end_time - (days * 24 * 60 * 60)
    
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history&includeAdjustedClose=true"
    # Actually, let's correct the variable name
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history&includeAdjustedClose=true"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"[ERROR] Direct download failed with status: {response.status_code}")
            return None
            
        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        
        # Standardize for our pipeline
        # Yahoo CSV usually has: Date, Open, High, Low, Close, Adj Close, Volume
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
            
        # Ensure we have 'price' (Close) and 'total_volume' (Volume)
        if 'close' in df.columns:
            df['price'] = df['close']
        elif 'adj close' in df.columns:
            df['price'] = df['adj close']
            
        if 'volume' in df.columns:
            df['total_volume'] = df['volume']
            
        df['market_cap'] = 0.0 # Placeholder
        
        # Filter and Save
        if len(df) > days:
            df = df.iloc[-days:]
            
        os.makedirs("data/raw", exist_ok=True)
        filename = f"data/raw/{coin_id}_prices.csv"
        df.to_csv(filename, index=False)
        print(f"[SUCCESS] Direct download saved {len(df)} rows to {filename}")
        return filename

    except Exception as e:
        print(f"[ERROR] Direct download completely failed: {e}")
        return None
