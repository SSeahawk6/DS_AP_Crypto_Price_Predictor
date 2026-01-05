import argparse
import pandas as pd
from src.api_collector import fetch_crypto_data
from src.feature_engineer import add_technical_indicators
from src.ml_model import train_model
from src.optimization import ModelOptimizer  # <--- NEW IMPORT
from src.backtester import Backtester
from src.utils import setup_environment     # <--- NEW IMPORT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, default="bitcoin")
    parser.add_argument("--days", type=int, default=1000)
    args = parser.parse_args()

    # 1. Setup
    setup_environment() # <--- NEW USAGE

    # 2. Data Collection
    csv_path = fetch_crypto_data(args.coins, args.days)
    if not csv_path:
        return

    # 3. Feature Engineering
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_features = add_technical_indicators(df)
    if df_features.empty:
        print("[ERROR] No data after processing.")
        return

    # 4. Optimization & Training
    print("--- Starting Optimization ---")
    X = df_features.drop(['target', 'next_day_return', 'price'], axis=1)
    y = df_features['target']
    
    # Use the new Optimizer to find the best model
    optimizer = ModelOptimizer(n_splits=3)
    model = optimizer.optimize_random_forest(X, y, n_iter=5)
    
    # Train the best model on the full dataset (Split happens inside train_model usually, 
    # but here we pass the optimized model to be fit)
    # For simplicity in this project structure, we just fit the best model:
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # 5. Backtesting
    print("--- Starting Backtest ---")
    test_df = df_features.iloc[split:].copy()
    test_df['prediction'] = predictions
    
    backtester = Backtester(initial_balance=10000)
    backtester.run(test_df)
    metrics = backtester.calculate_metrics()
    
    print(f"[RESULT] Backtest Metrics: {metrics}")
    backtester.plot_results(test_df, filename=f"results/chart_{args.coins}.png")

if __name__ == "__main__":
    main()
