import pandas as pd
import numpy as np

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Force Price to Numeric (Safety Check)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    
    if len(df) < 30:
        return df 
        
    # 2. Add Indicators
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['std_20'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
    df['rsi'] = calculate_rsi(df)
    
    # 3. Add Lag Features
    for lag in [1, 2, 3, 7]:
        df[f'return_lag_{lag}'] = df['price'].pct_change().shift(lag)
    
    # 4. Create Target (1 if Price Up > 0.5%, 0 otherwise)
    # This "Picky Trader" logic avoids trading on noise or tiny gains typically eaten by fees.
    df['next_day_return'] = df['price'].pct_change().shift(-1)
    df['target'] = (df['next_day_return'] > 0.005).astype(int)
    
    # 5. Drop NaNs created by windows
    df.dropna(inplace=True)
    
    return df
