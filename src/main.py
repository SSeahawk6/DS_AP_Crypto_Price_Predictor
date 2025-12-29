import argparse
import sys
import os
import concurrent.futures
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_collector import fetch_crypto_data
from src.feature_engineer import add_technical_indicators
from src.ml_model import train_model
from src.dl_model import train_dl_model  # <--- NEW IMPORT
from src.backtester import Backtester

def process_coin(coin_id, days, model_type):
    """
    Worker function to process a single coin with a specific model.
    """
    print(f"[{coin_id.upper()}] Starting analysis using {model_type}...")
    
    # 1. Fetch
    csv_path = fetch_crypto_data(coin_id, days)
    if not csv_path:
        return f"[{coin_id.upper()}] Failed data."

    # 2. Features
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=True, index_col='timestamp')
    df_features = add_technical_indicators(df)
    
    # 3. Train (Switch based on user input)
    if model_type == 'neural_net':
        model, X_test, y_test, predictions = train_dl_model(df_features)
        # Re-map X_test to DataFrame for backtester
        # (Neural net converts to numpy, so we need to get indices back)
        split_point = int(len(df_features) * 0.8)
        test_prices = df_features.iloc[split_point:]['price']
    else:
        # Default to Random Forest
        model, X_test, y_test, predictions = train_model(df_features)
        test_prices = df_features.loc[X_test.index, 'price']
    
    # 4. Backtest
    backtester = Backtester(initial_balance=10000)
    final_balance = backtester.run(test_prices, predictions)
    
    # 5. Metrics
    metrics = backtester.calculate_metrics()
    
    # Save Plot
    chart_filename = f"results/chart_{coin_id}_{model_type}.png"
    backtester.plot_results(test_prices, filename=chart_filename)
    
    result_msg = (
        f"[{coin_id.upper()}] ({model_type}) Finished.\n"
        f"   Balance: ${final_balance:.2f}\n"
        f"   Return: {metrics['Total Return']}\n"
        f"   Sharpe Ratio: {metrics['Sharpe Ratio']}\n"
        f"   Max Drawdown: {metrics['Max Drawdown']}"
    )
    return result_msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=["bitcoin"], help="List of coins")
    parser.add_argument("--days", type=int, default=365, help="Days of data")
    # NEW ARGUMENT:
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "neural_net"], help="Model type: rf (Random Forest) or neural_net")
    
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    start_time = time.time()
    print(f"--- Starting Analysis with Model: {args.model.upper()} ---")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_coin, coin, args.days, args.model) for coin in args.coins]
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
            print("-" * 30)
            
    print(f"--- Total Time: {time.time() - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
