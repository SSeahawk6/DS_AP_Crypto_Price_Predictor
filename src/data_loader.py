# This module acts as an API Collector to fetch data from Yahoo Finance.

import pandas as pd
import os

def fetch_crypto_data(coin_id, days):
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
        print(f"[ERROR] Failed to fetch data: {e}")
        return None
