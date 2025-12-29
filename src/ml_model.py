import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_model(df):
    """
    Trains a Random Forest model using GridSearchCV to find best parameters.
    Matches Lecture 12b: Hyperparameter Tuning.
    """
    # 1. Define Features
    features = ['sma_20', 'bb_upper', 'bb_lower', 'rsi', 'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_7']
    
    X = df[features]
    
    # 2. Time-Based Split (80/20)
    split_point = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"[INFO] Training on {len(X_train)} rows...")
    
    # 3. Hyperparameter Tuning (Lecture 12b)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"[INFO] Best Parameters found: {grid_search.best_params_}")
    
    # 4. Evaluate
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[RESULT] Model Accuracy on Test Set: {accuracy:.2%}")
    
    return best_model, X_test, y_test, predictions
