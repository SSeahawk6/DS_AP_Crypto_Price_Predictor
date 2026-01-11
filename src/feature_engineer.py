import pandas as pd
import numpy as np

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df: pd.DataFrame, threshold: float = 0.005) -> pd.DataFrame:
    df = df.copy()
    
    # Force Price to Numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    
    if len(df) < 30:
        return df 
        
    # Indicators
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['sma_50'] = df['price'].rolling(window=50).mean()
    df['std_20'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
    df['rsi'] = calculate_rsi(df)
    
    # Lag Features
    for lag in [1, 2, 3, 7]:
        df[f'return_lag_{lag}'] = df['price'].pct_change().shift(lag)
    
    # Target (1 if Price Up > threshold, 0 otherwise)
    # The threshold prevents trading on insignificant moves (noise).
    df['next_day_return'] = df['price'].pct_change().shift(-1)
    df['target'] = (df['next_day_return'] > threshold).astype(int)
    
    df.dropna(inplace=True)
    
    return df
