import pandas as pd
import numpy as np

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators, Lag Features (Memory), and a threshold-based Target.
    """
    df = df.copy()
    
    # --- 1. Basic Indicators ---
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['std_20'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
    df['rsi'] = calculate_rsi(df)
    
    # --- 2. Advanced: Lag Features (Memory) ---
    # This gives the model context: "What happened yesterday?"
    # We use pct_change() so the model sees returns (%), not raw price ($)
    for lag in [1, 2, 3, 7]:
        df[f'return_lag_{lag}'] = df['price'].pct_change().shift(lag)
    
    # --- 3. The TA's Requirement: Threshold Target ---
    # We calculate the NEXT day's return
    df['next_day_return'] = df['price'].pct_change().shift(-1)
    
    # STRICT RULE: Label 1 if return > 0.5% (0.005), else 0
    # This matches the TA's explicit example in the screenshot.
    df['target'] = (df['next_day_return'] > 0.005).astype(int)
    
    # Drop NaNs created by rolling windows and lags
    df.dropna(inplace=True)
    
    return df