import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

class ModelOptimizer:
    def __init__(self, n_splits=5, random_state=42):
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        self.random_state = random_state

    def optimize_random_forest(self, X, y, n_iter=10):
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
