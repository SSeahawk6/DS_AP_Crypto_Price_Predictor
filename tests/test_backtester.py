import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.backtester import Backtester

class TestBacktester:
    
    @pytest.fixture
    def sample_data(self):
        """
        Creates a predictable market scenario.
        Price: 100 -> 110 -> 120 -> 90
        Strategy: Buy -> Hold -> Sell -> Wait
        """
        dates = pd.date_range('2023-01-01', periods=4)
        df = pd.DataFrame({
            'price': [100.0, 110.0, 120.0, 90.0],
            # 1 = Buy, 0 = Sell
            'prediction': [1, 1, 0, 0] 
        }, index=dates)
        return df

    def test_initialization(self):
        """Test if the wallet starts with the correct amount."""
        bt = Backtester(initial_balance=5000)
        assert bt.initial_balance == 5000
        assert bt.cash == 5000
        assert bt.holdings == 0
        assert bt.history == []

    def test_buy_logic(self):
        """Test converting cash to crypto."""
        bt = Backtester(initial_balance=1000)
        price = 100
        
        bt.buy(price)
        
        assert bt.cash == 0
        assert bt.holdings == 10.0  # 1000 / 100 = 10 units
        
        # Test Double Buy: Should not be able to buy again with 0 cash
        bt.buy(price) 
        assert bt.holdings == 10.0 

    def test_sell_logic(self):
        """Test converting crypto back to cash."""
        bt = Backtester(initial_balance=1000)
        
        # Manually set state to simulate owning crypto
        bt.cash = 0
        bt.holdings = 10  # We own 10 coins
        
        price = 200 # Price doubled!
        bt.sell(price)
        
        assert bt.holdings == 0
        assert bt.cash == 2000.0 # 10 coins * $200
        
        # Test Double Sell: Should not be able to sell 0 coins
        bt.sell(price)
        assert bt.cash == 2000.0

    def test_run_simulation(self, sample_data):
        """
        Tests the full loop 'run' method.
        Scenario:
        1. Buy at $100 (Holdings=10, Cash=0) -> Value $1000
        2. Hold at $110 (Holdings=10, Cash=0) -> Value $1100
        3. Sell at $120 (Holdings=0, Cash=1200) -> Value $1200
        4. Wait at $90  (Holdings=0, Cash=1200) -> Value $1200
        """
        bt = Backtester(initial_balance=1000)
        results = bt.run(sample_data)
        
        # Check Final Value
        final_value = results.iloc[-1]['value']
        assert final_value == 1200.0
        
        # Check that history length matches data length
        assert len(results) == 4

    def test_calculate_metrics(self, sample_data):
        """Ensures math metrics return the correct dictionary format."""
        bt = Backtester(initial_balance=1000)
        bt.run(sample_data)
        
        metrics = bt.calculate_metrics()
        
        assert "Total Return" in metrics
        assert "Sharpe Ratio" in metrics
        assert "Max Drawdown" in metrics
        
        # In our scenario, we made $200 on $1000 = 20%
        # NOTE: This assertion will verify if your src/backtester.py fix is working
        assert metrics['Total Return'] == "20.00%"

    @patch('src.backtester.plt')  # Mock matplotlib
    @patch('src.backtester.os.makedirs') # Mock creating folders
    def test_plot_results(self, mock_makedirs, mock_plt, sample_data):
        """Test that plotting code runs without errors (doesn't actually draw)."""
        bt = Backtester(initial_balance=1000)
        bt.run(sample_data)
        
        bt.plot_results(sample_data, filename="results/test_chart.png")
        
        # verify the plot function was called
        mock_plt.figure.assert_called()
        mock_plt.savefig.assert_called_with("results/test_chart.png")
