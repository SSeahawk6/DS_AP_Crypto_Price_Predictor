import argparse
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_collector import fetch_crypto_data
from src.feature_engineer import add_technical_indicators
from src.ml_model import train_model
from src.backtester import Backtester

def main():
    parser = argparse.ArgumentParser(description="Crypto Price Predictor ML Pipeline")
    parser.add_argument("--coin", type=str, default="bitcoin", help="Cryptocurrency ID (e.g., bitcoin)")
    parser.add_argument("--days", type=int, default=365, help="Number of days of data to fetch")
    
    args = parser.parse_args()
    
    print(f"--- Starting Analysis for {args.coin.upper()} ---")
    
    # Data Collection
    csv_path = fetch_crypto_data(args.coin, args.days)
    if not csv_path:
        print("Exiting due to data fetching error.")
        return

    # Load and Engineer Features
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=True, index_col='timestamp')
    
    print("[INFO] Engineering technical features...")
    df_features = add_technical_indicators(df)
    
    # Train Model
    print("[INFO] Training Random Forest Model...")
    model, X_test, y_test, predictions = train_model(df_features)
    
    # Run Backtest
    print("[INFO] Running Backtest Simulation...")
    test_prices = df_features.loc[X_test.index, 'price']
    
    backtester = Backtester(initial_balance=10000)
    final_balance = backtester.run(test_prices, predictions)
    
    # Report Results
    return_pct = ((final_balance - 10000) / 10000) * 100
    print("-" * 30)
    print(f"FINAL RESULT: ${final_balance:.2f} ({return_pct:+.2f}%)")
    print("-" * 30)
    
    backtester.plot_results(test_prices)

if __name__ == "__main__":
    main()