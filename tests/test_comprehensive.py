import unittest
import pandas as pd
import numpy as np
import os
import shutil
from src.models import ModelOptimizer
from src.utils import setup_environment, ensure_directories_exist
from src.evaluation import Backtester
from src.feature_engineer import add_technical_indicators
from src.evaluation import plot_price_history, plot_technical_indicators

class TestEndToEnd(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Run once before all tests."""
        setup_environment(seed=42)
        cls.df = pd.DataFrame({
            'price': np.random.uniform(20000, 30000, 100),
            'volume': np.random.uniform(100, 500, 100)
        })
        cls.df.index = pd.date_range(start='2024-01-01', periods=100)
        cls.df.index.name = 'timestamp'
        cls.processed_df = add_technical_indicators(cls.df)

    def test_utils(self):
        """Test utils module (coverage boost)."""
        if os.path.exists('results'): shutil.rmtree('results')
        ensure_directories_exist()
        self.assertTrue(os.path.exists('results'))

    def test_utils(self):
        """Test utils module (coverage boost)."""
        if os.path.exists('results'): shutil.rmtree('results')
        
        ensure_directories_exist(['results', 'data']) 
        
        self.assertTrue(os.path.exists('results'))

    def test_visualization(self):
        """Test visualization module (coverage boost)."""
        # We just check if the function runs without error
        try:
            plot_price_history(self.df, 'test_coin')
            plot_technical_indicators(self.processed_df, 'test_coin')
        except Exception as e:
            self.fail(f"Visualization raised exception: {e}")
            
    def test_backtest_math(self):
        """Verify money math."""
        bt = Backtester(1000)
        bt.buy(100)
        self.assertEqual(bt.cash, 0)
        bt.sell(200)
        self.assertEqual(bt.cash, 2000)

if __name__ == '__main__':
    unittest.main()
