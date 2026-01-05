import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineer import add_technical_indicators

def test_technical_indicators_exist():
    df = pd.DataFrame({'price': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] * 5})
    df_result = add_technical_indicators(df)
    assert 'rsi' in df_result.columns
    assert 'target' in df_result.columns
