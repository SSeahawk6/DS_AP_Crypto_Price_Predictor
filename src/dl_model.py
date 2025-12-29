import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def train_dl_model(df):
    """
    Trains a Deep Learning model (Neural Network) for price prediction.
    Architecture: Input -> Dense(64) -> Dropout -> Dense(32) -> Output
    """
    # 1. Define Features (Must match what we created in feature_engineer.py)
    features = [
        'sma_20', 'bb_upper', 'bb_lower', 'rsi', 
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_7'
    ]
    
    # Filter to ensure features exist
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].values
    y = df['target'].values
    
    # 2. Scale the Data (CRITICAL for Neural Networks)
    # Neural Nets fail if inputs are not between 0 and 1 (or -1 and 1)
    scaler = StandardScaler()
    
    # Time-based split
    split = int(len(X) * 0.8)
    X_train = scaler.fit_transform(X[:split])
    X_test = scaler.transform(X[split:])
    y_train, y_test = y[:split], y[split:]
    
    print(f"[INFO] Training Neural Network on {len(X_train)} rows...")
    
    # 3. Build the Neural Network (Lecture 10 architecture)
    model = Sequential([
        # Layer 1: 64 Neurons, ReLU activation
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        
        # Dropout: Randomly turn off 20% of neurons to prevent overfitting
        Dropout(0.2),
        
        # Layer 2: 32 Neurons
        Dense(32, activation='relu'),
        
        # Output Layer: 1 Neuron (Probability between 0 and 1)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # 4. Train
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    
    # 5. Predict
    # The NN outputs probabilities (e.g., 0.75). We convert to 0 or 1.
    probs = model.predict(X_test, verbose=0)
    predictions = (probs > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, predictions)
    print(f"[RESULT] Neural Net Accuracy: {acc:.2%}")
    
    return model, X[split:], y[split:], predictions
