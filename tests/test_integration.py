import unittest
import pandas as pd
import numpy as np
from src.models import train_rf_model
from src.evaluation import Backtester

def test_full_pipeline():
    # Create Dummy Data to run an integration test. 
    dates = pd.date_range(start='2023-01-01', periods=50)
    df = pd.DataFrame({
        'price': [100 + i for i in range(50)],
        'sma_20': [100] * 50,
        'bb_upper': [110] * 50,
        'bb_lower': [90] * 50,
        'rsi': [50] * 50,
        'target': [1] * 50 # Always goes up to make sure the backtester buys and holds as it should
    }, index=dates)
    
    # Train Model
    model, X_test, y_test, preds = train_rf_model(df)
    
    # Prepare Backtest Data to align training and testing data to guarantee length matches between the original dataframe and the predictions.
    test_df = df.iloc[-len(preds):].copy()
    test_df['prediction'] = preds
    
    backtester = Backtester(initial_balance=1000)
    results = backtester.run(test_df)
    
    # Assertions
    assert not results.empty
    assert 'value' in results.columns
    # Since target is always 1, we should have bought and held
    assert backtester.holdings > 0 or backtester.cash > 1000
