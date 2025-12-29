import pandas as pd
import numpy as np

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    """
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Technical Indicators (RSI, SMA, Bollinger Bands) to the DataFrame.
    """
    df = df.copy()
    
    # 1. Simple Moving Average (20 days)
    df['sma_20'] = df['price'].rolling(window=20).mean()
    
    # 2. Bollinger Bands (20 days, 2 std dev)
    rolling_std = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (rolling_std * 2)
    df['bb_lower'] = df['sma_20'] - (rolling_std * 2)
    
    # 3. Relative Strength Index (RSI)
    df['rsi'] = calculate_rsi(df)
    
    # 4. Target Variable: Will the price be higher tomorrow? (1 = Yes, 0 = No)
    # We shift(-1) to compare 'today's price' with 'tomorrow's price'
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    
    # Drop rows with NaN values created by rolling windows
    df.dropna(inplace=True)
    
    return df
    