import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.api_collector import fetch_crypto_data


import sys

class TestAPICollector:

    @pytest.fixture(autouse=True)
    def mock_yf_download(self):
        """Mocks the yfinance module globally for all tests to avoid environment issues."""
        mock_yf = MagicMock()
        mock_download = MagicMock()
        mock_yf.download = mock_download
        
        with patch.dict(sys.modules, {'yfinance': mock_yf}):
            yield mock_download

    @pytest.fixture
    def mock_yf_data(self):
        """Creates a fake DataFrame similar to what yfinance returns."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        columns = pd.MultiIndex.from_product([['Close', 'Volume', 'Open', 'High', 'Low'], ['BTC-USD']])
        data = pd.DataFrame(np.random.rand(10, 5), index=dates, columns=columns)
        return data

    @patch('src.api_collector.pd.DataFrame.to_csv')
    def test_fetch_success(self, mock_to_csv, mock_yf_download, mock_yf_data):
        """Test happy path: Valid coin, API returns data, file 'saved'."""
        # 1. Setup the Mock
        mock_yf_download.return_value = mock_yf_data
        
        # 2. Call the function
        # Ensure we don't hit the cache
        with patch('src.api_collector.os.path.exists', return_value=False):
            result_filename = fetch_crypto_data("bitcoin", days=5)
        
        # 3. Assertions
        assert result_filename == "data/bitcoin_prices.csv"
        mock_yf_download.assert_called_once()
        args, kwargs = mock_yf_download.call_args
        assert args[0] == "BTC-USD"
        mock_to_csv.assert_called_once()

    def test_fetch_invalid_coin(self, mock_yf_download):
        """Test that an unsupported coin returns None immediately."""
        result = fetch_crypto_data("unsupported_coin_xyz", days=5)
        assert result is None
        mock_yf_download.assert_not_called()

    def test_fetch_api_failure(self, mock_yf_download):
        """Test how the function handles a crash in yfinance."""
        mock_yf_download.side_effect = Exception("API connection broken")
        
        with patch('src.api_collector.os.path.exists', return_value=False):
            result = fetch_crypto_data("bitcoin", days=5)
        
        assert result is None

    @patch('src.api_collector.os.path.exists')
    def test_fetch_cached_data(self, mock_exists, mock_yf_download):
        """Test that data is NOT re-downloaded if it exists."""
        mock_exists.return_value = True
        result = fetch_crypto_data("bitcoin", days=5)
        assert result == "data/bitcoin_prices.csv"
        mock_yf_download.assert_not_called()