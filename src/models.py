# This module defines Machine Learning and Deep Learning models.

from typing import Tuple, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    print("[WARN] TensorFlow not found. Deep Learning model will be skipped.")
    TF_AVAILABLE = False

# Random Forest Model


def train_rf_model(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.Series, np.ndarray]:
    """
    Trains a Random Forest model using RandomizedSearchCV.
    Includes technical indicators and lag features.
    """
    # Feature selection
    features = [
        'sma_20', 'bb_upper', 'bb_lower', 'rsi', 
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_7'
    ]
    
    # Safety check: ensure these columns actually exist
    # We filter features to prevent the model from crashing if a column failed to generate
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < len(features):
        print(f"[WARNING] Some features are missing! Using only: {available_features}")
    
    X = df[available_features]
    y = df['target']
    
    X = df[available_features]
    y = df['target']
    
    # Time-based split (80/20) to prevent look-ahead bias
    split_point = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"[INFO] Training on {len(X_train)} rows with {len(available_features)} features...")
    
    # Search space for RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"[INFO] Best Parameters found: {grid_search.best_params_}")
    
    # Evaluate
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[RESULT] Model Accuracy on Test Set: {accuracy:.2%}")
    
    return best_model, X_test, y_test, predictions

class ModelOptimizer:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        self.random_state = random_state

    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 10) -> Any:
        """
        Finds the best hyperparameters using Randomized Search.
        """
        rf = RandomForestClassifier(random_state=self.random_state)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state
        )
        
        search.fit(X, y)
        print(f"[INFO] Best Params: {search.best_params_}")
        print(f"[INFO] Best Accuracy: {search.best_score_:.4f}")
        
        return search.best_estimator_


# Deep Learning Model

def train_dl_model(df: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
    """
    Trains a Deep Learning model (Neural Network) for price prediction.
    Architecture: Input -> Dense(64) -> Dropout -> Dense(32) -> Output
    """
    if not TF_AVAILABLE:
        print("[INFO] Skipping Deep Learning training (TensorFlow missing).")
        # Expected: model, X_test, y_test, predictions
        return None, None, None, []
    # Define Features (Must match what we created in feature_engineer.py)
    features = [
        'sma_20', 'bb_upper', 'bb_lower', 'rsi', 
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_7'
    ]
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].values
    y = df['target'].values
    

    # Scale data (standardization is critical for NN convergence)
    scaler = StandardScaler()
    
    # Time-based split: 80% Train, 20% Test
    split = int(len(X) * 0.8)
    X_train = scaler.fit_transform(X[:split])
    X_test = scaler.transform(X[split:])
    y_train, y_test = y[:split], y[split:]
    
    print(f"[INFO] Training Neural Network on {len(X_train)} rows...")
    
    # Architecture: 2 hidden layers with Dropout for regularization
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Train with Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=16, 
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict probabilities (0-1) and threshold at 0.5
    probs = model.predict(X_test, verbose=0)
    predictions = (probs > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, predictions)
    print(f"[RESULT] Neural Net Accuracy: {acc:.2%}")
    
    return model, X[split:], y[split:], predictions
