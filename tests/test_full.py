import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in the path for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineer import add_technical_indicators, calculate_rsi
from src.backtester import Backtester
from src.ml_model import train_model

class TestProjectLogic(unittest.TestCase):
    def setUp(self):
        """Create high-quality synthetic data for testing."""
        np.random.seed(42) # Fulfills Reproducibility requirement
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.df = pd.DataFrame({
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        self.df.index.name = 'timestamp'

    def test_indicator_math(self):
        """Verify that technical indicators are within logical bounds."""
        processed = add_technical_indicators(self.df)
        self.assertTrue((processed['rsi'] <= 100).all())
        self.assertTrue((processed['rsi'] >= 0).all())
        self.assertIn('bb_upper', processed.columns)

    def test_target_generation(self):
        """Ensure the target variable is correctly classified."""
        processed = add_technical_indicators(self.df)
        self.assertTrue(set(processed['target'].unique()).issubset({0, 1}))

    def test_backtester_initialization(self):
        """Verify the simulator starts with correct capital."""
        bt = Backtester(initial_balance=5000)
        self.assertEqual(bt.cash, 5000)
        self.assertEqual(bt.holdings, 0)

    def test_trading_execution(self):
        """Simulate a buy and sell cycle to verify arithmetic."""
        bt = Backtester(initial_balance=1000)
        bt.buy(100, '2023-01-01')
        self.assertEqual(bt.holdings, 10)
        bt.sell(150, '2023-01-02')
        self.assertEqual(bt.cash, 1500)

    def test_model_training_output(self):
        """Verify ML model returns required performance objects."""
        # Simple data for training test
        train_df = pd.DataFrame(np.random.rand(100, 4), columns=['rsi', 'sma_20', 'bb_upper', 'bb_lower'])
        train_df['target'] = (train_df['rsi'] > 0.5).astype(int)
        
        model, X_test, y_test, preds = train_model(train_df)
        self.assertIsNotNone(model)
        self.assertEqual(len(y_test), len(preds))

if __name__ == '__main__':
    unittest.main()
