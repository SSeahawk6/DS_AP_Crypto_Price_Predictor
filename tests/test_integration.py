import unittest
import pandas as pd
import numpy as np
from src.ml_model import train_model
from src.backtester import Backtester

def test_full_pipeline():
    # 1. Create Dummy Data
    dates = pd.date_range(start='2023-01-01', periods=50)
    df = pd.DataFrame({
        'price': [100 + i for i in range(50)],
        'sma_20': [100] * 50,
        'bb_upper': [110] * 50,
        'bb_lower': [90] * 50,
        'rsi': [50] * 50,
        'target': [1] * 50 # Always goes up
    }, index=dates)
    
    # 2. Train Model
    model, X_test, y_test, preds = train_model(df)
    
    # 3. Prepare Backtest Data (The Fix)
    # The new Backtester expects a DataFrame with a 'prediction' column
    test_df = df.iloc[-len(preds):].copy()
    test_df['prediction'] = preds
    
    # 4. Run Backtest
    backtester = Backtester(initial_balance=1000)
    results = backtester.run(test_df)
    
    # Assertions
    assert not results.empty
    assert 'value' in results.columns
    # Since target is always 1, we should have bought and held
    assert backtester.holdings > 0 or backtester.cash > 1000
