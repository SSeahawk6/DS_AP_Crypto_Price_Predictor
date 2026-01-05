import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.api_collector import fetch_crypto_data

class TestAPICollector:

    @pytest.fixture
    def mock_yf_data(self):
        """Creates a fake DataFrame similar to what yfinance returns."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        
        # Create a DataFrame with a MultiIndex to simulate yfinance's complex structure
        # Structure: (Price Type, Ticker) -> ('Close', 'BTC-USD')
        columns = pd.MultiIndex.from_product([['Close', 'Volume', 'Open', 'High', 'Low'], ['BTC-USD']])
        
        # Create random data
        data = pd.DataFrame(np.random.rand(10, 5), index=dates, columns=columns)
        
        # Rename levels to look like yfinance raw output if needed, 
        # but for the test, simply matching the structure is enough.
        # The code logic expects tuple columns or checks isinstance(c, tuple)
        
        return data

    @patch('src.api_collector.yf.download')
    @patch('src.api_collector.pd.DataFrame.to_csv')  # Mock to_csv so we don't create real files
    def test_fetch_success(self, mock_to_csv, mock_yf_download, mock_yf_data):
        """Test happy path: Valid coin, API returns data, file 'saved'."""
        
        # 1. Setup the Mock
        # When yf.download is called, return our fake dataframe
        mock_yf_download.return_value = mock_yf_data
        
        # 2. Call the function
        result_filename = fetch_crypto_data("bitcoin", days=5)
        
        # 3. Assertions
        assert result_filename == "data/bitcoin_prices.csv"
        
        # Verify yfinance was called with correct ticker
        mock_yf_download.assert_called_once()
        args, kwargs = mock_yf_download.call_args
        assert args[0] == "BTC-USD" # Ensures mapping (bitcoin -> BTC-USD) worked
        
        # Verify to_csv was called (meaning logic reached the end)
        mock_to_csv.assert_called_once()

    @patch('src.api_collector.yf.download')
    def test_fetch_invalid_coin(self, mock_yf_download):
        """Test that an unsupported coin returns None immediately."""
        
        result = fetch_crypto_data("unsupported_coin_xyz", days=5)
        
        assert result is None
        mock_yf_download.assert_not_called() # Should fail before downloading

    @patch('src.api_collector.yf.download')
    def test_fetch_api_failure(self, mock_yf_download):
        """Test how the function handles a crash in yfinance."""
        
        # 1. Setup Mock to explode
        mock_yf_download.side_effect = Exception("API connection broken")
        
        # 2. Call function
        result = fetch_crypto_data("bitcoin", days=5)
        
        # 3. Assertions
        assert result is None