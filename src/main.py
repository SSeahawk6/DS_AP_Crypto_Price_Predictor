import argparse
import sys
import os
import concurrent.futures
import time

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_collector import fetch_crypto_data
from src.feature_engineer import add_technical_indicators
from src.ml_model import train_model
from src.backtester import Backtester

def process_coin(coin_id, days):
    """
    Worker function to process a single coin.
    """
    print(f"[{coin_id.upper()}] Starting analysis...")
    
    # 1. Fetch Data
    csv_path = fetch_crypto_data(coin_id, days)
    if not csv_path:
        return f"[{coin_id.upper()}] Failed to fetch data."

    # 2. Engineer Features
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=True, index_col='timestamp')
    df_features = add_technical_indicators(df)
    
    # 3. Train Model
    model, X_test, y_test, predictions = train_model(df_features)
    
    # 4. Backtest
    test_prices = df_features.loc[X_test.index, 'price']
    backtester = Backtester(initial_balance=10000)
    final_balance = backtester.run(test_prices, predictions)
    
    # 5. Save Results & Metrics
    metrics = backtester.calculate_metrics()
    
    # Save Plot
    chart_filename = f"results/chart_{coin_id}.png"
    backtester.plot_results(test_prices, filename=chart_filename)
    
    # Format the success message with the new metrics
    result_msg = (
        f"[{coin_id.upper()}] Finished.\n"
        f"   Balance: ${final_balance:.2f}\n"
        f"   Return: {metrics['Total Return']}\n"
        f"   Sharpe Ratio: {metrics['Sharpe Ratio']}\n"
        f"   Max Drawdown: {metrics['Max Drawdown']}"
    )
    return result_msg

def main():
    parser = argparse.ArgumentParser(description="Parallel Crypto Price Predictor")
    parser.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum", "solana"], help="List of coins to analyze")
    parser.add_argument("--days", type=int, default=365, help="Days of data to fetch")
    
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    start_time = time.time()
    print(f"--- Starting Parallel Analysis for: {args.coins} ---")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_coin, coin, args.days) for coin in args.coins]
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
            print("-" * 30)
            
    print(f"--- Total Time: {time.time() - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
