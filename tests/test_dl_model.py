import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from src.dl_model import train_dl_model
import os

# Turn off noisy TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TestDLModel:
    
    @pytest.fixture
    def mock_df(self):
        """
        Creates a small, safe dataset (50 rows) to test the neural network 
        without waiting for a long training session.
        """
        n_rows = 50
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_rows),
            'target': np.random.randint(0, 2, n_rows), # Binary target (0 or 1)
            
            # Feature columns expected by your model
            'sma_20': np.random.rand(n_rows) * 100,
            'bb_upper': np.random.rand(n_rows) * 100,
            'bb_lower': np.random.rand(n_rows) * 100,
            'rsi': np.random.rand(n_rows) * 100,
            'return_lag_1': np.random.randn(n_rows),
            'return_lag_2': np.random.randn(n_rows),
            'return_lag_3': np.random.randn(n_rows),
            'return_lag_7': np.random.randn(n_rows),
        }
        return pd.DataFrame(data)

    def test_dl_model_runs_end_to_end(self, mock_df):
        """
        Tests the training pipeline.
        Verifies:
        1. Model builds correctly (Keras object).
        2. Returns the correct data split shapes.
        3. Predictions are binary (0 or 1).
        """
        model, X_test, y_test, predictions = train_dl_model(mock_df)
        
        # 1. Check Model Architecture
        assert isinstance(model, tf.keras.Model)
        # We expect 4 layers: Dense(64) -> Dropout -> Dense(32) -> Output(1)
        assert len(model.layers) == 4
        
        # 2. Check Data Shapes (50 rows * 0.2 split = 10 test rows)
        expected_test_size = 10 
        assert len(X_test) == expected_test_size
        assert len(y_test) == expected_test_size
        assert len(predictions) == expected_test_size
        
        # 3. Check Prediction Validity
        # Predictions must be integers 0 or 1 (not probabilities like 0.75)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_dl_model_missing_features(self, mock_df):
        """
        Ensures the code is robust:
        If 'rsi' column is missing, the model should still train on the others.
        """
        incomplete_df = mock_df.drop(columns=['rsi'])
        
        model, X_test, y_test, preds = train_dl_model(incomplete_df)
        
        assert model is not None
        # Check that the input layer accepted 7 features instead of 8
        input_shape = model.layers[0].input_shape
        assert input_shape[1] == 7
