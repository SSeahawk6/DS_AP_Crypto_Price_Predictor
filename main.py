import argparse
import pandas as pd
from src.data_loader import fetch_crypto_data
from src.feature_engineer import add_technical_indicators
from src.models import train_rf_model, ModelOptimizer, train_dl_model
from src.evaluation import Backtester, plot_feature_importance, plot_strategy_comparison
from src.utils import setup_environment

def process_coin(coin, days, threshold, initial_balance=10000):
    """
    Runs the full pipeline for a single coin.
    Returns a dictionary of metrics.
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING: {coin.upper()} | Threshold={threshold}")
    print(f"{'='*50}")
    
    # Data Collection
    csv_path = fetch_crypto_data(coin, days)
    if not csv_path:
        return None

    # Feature Engineering
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_features = add_technical_indicators(df, threshold=threshold)
    if df_features.empty:
        print("[ERROR] No data after processing.")
        return None

    # Prepare Data
    X = df_features.drop(['target', 'next_day_return', 'price'], axis=1)
    y = df_features['target']
    
    # Time-based split: 80% train / 20% test. X_test is locked until final evaluation to prevent data leakage.
    split = int(len(X) * 0.8)
    
    X_train = X.iloc[:split] # Used for Optimization + Training
    y_train = y.iloc[:split]
    
    X_test = X.iloc[split:] # Used ONLY for final evaluation
    y_test = y.iloc[split:]

    # Model Comparison (Random Forest vs Deep Learning)
    
    # --- Random Forest ---
    print("\n[MODEL A] Random Forest Optimization...")
    # Optimize ONLY on the Train set

    optimizer = ModelOptimizer(n_splits=3)
    rf_model = optimizer.optimize_random_forest(X_train, y_train, n_iter=5)
    
    # Refit on the full Train set before predicting on Test
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # Feature Importance
    print("[INFO] Generating Feature Importance Chart...")
    plot_feature_importance(rf_model, X.columns, filename=f"results/feature_importance_{coin}.png")

    # --- Deep Learning ---
    print("\n[MODEL B] Deep Learning Training...")

    try:
        dl_model, _, _, dl_preds = train_dl_model(df_features)
    except Exception as e:
        print(f"[WARN] DL Model Failed: {e}")
        dl_preds = []

    # --- Random Forest Strategy ---
    print("\n--- Starting Backtest (Random Forest Strategy) ---")
    
    # Align predictions with test set index
    test_df = df_features.iloc[split:].copy()
    
    if len(rf_preds) != len(test_df):
        print(f"[WARNING] Length mismatch: Preds {len(rf_preds)} vs Test DF {len(test_df)}")
        test_df = test_df.iloc[-len(rf_preds):]
        
    test_df['prediction'] = rf_preds
    
    rf_backtester = Backtester(initial_balance=initial_balance)
    rf_backtester.run(test_df)
    rf_metrics = rf_backtester.calculate_metrics()
    
    print(f"[RESULT] {coin.capitalize()} Metrics: {rf_metrics}")
    rf_backtester.plot_results(test_df, filename=f"results/chart_{coin}.png")

    # --- Benchmark: Buy & Hold ---
    print("--- Calculating Benchmark (Buy & Hold) ---")
    test_df_bh = test_df.copy()
    test_df_bh['prediction'] = 1  # Always Buy
    
    bh_backtester = Backtester(initial_balance=initial_balance)
    bh_backtester.run(test_df_bh)
    bh_metrics = bh_backtester.calculate_metrics()
    print(f"[BENCHMARK] {coin.capitalize()} B&H Metrics: {bh_metrics}")

    # --- Benchmark: SMA Crossover ---
    print("--- Calculating Benchmark (SMA Crossover) ---")
    test_df_sma = test_df.copy()
    # Signal: 1 if SMA_20 > SMA_50 (Golden Cross)
    test_df_sma['prediction'] = (test_df_sma['sma_20'] > test_df_sma['sma_50']).astype(int)
    
    sma_backtester = Backtester(initial_balance=initial_balance)
    sma_backtester.run(test_df_sma)
    sma_metrics = sma_backtester.calculate_metrics()

    # --- Deep Learning Strategy ---
    dl_metrics = {"Total Return": "N/A", "Sharpe Ratio": "N/A", "Max Drawdown": "N/A"}
    if len(dl_preds) > 0:
        print("--- Calculating Deep Learning Metrics ---")
        test_df_dl = df_features.iloc[split:].copy()
        
        # Align lengths
        if len(dl_preds) != len(test_df_dl):
             test_df_dl = test_df_dl.iloc[-len(dl_preds):]
             
        test_df_dl['prediction'] = dl_preds
        
        dl_backtester = Backtester(initial_balance=initial_balance)
        dl_backtester.run(test_df_dl)
        dl_metrics = dl_backtester.calculate_metrics()
        print(f"[MODEL B] Deep Learning Metrics: {dl_metrics}")

    # Visualization
    
    # Strategy Comparison Chart
    strategies = {
        "Random Forest": rf_backtester.history,
        "Deep Learning": dl_backtester.history if len(dl_preds) > 0 else [],
        "Buy & Hold": bh_backtester.history,
        "SMA Crossover": sma_backtester.history
    }
    # Filter out empty strategies (e.g. if DL failed)
    strategies = {k: v for k, v in strategies.items() if len(v) > 0}
    
    plot_strategy_comparison(test_df, strategies, filename=f"results/comparison_{coin}.png")
    
    return {
        "coin": coin,
        "rf_metrics": rf_metrics,
        "dl_metrics": dl_metrics,
        "bh_metrics": bh_metrics,
        "sma_metrics": sma_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, default="all", 
                        help="Comma-separated list of coins or 'all' (default: all)")
    parser.add_argument("--days", type=int, default=1825, help="Number of days of data (default: 5 years)")
    parser.add_argument("--threshold", type=float, default=0.005, help="Target return threshold (default: 0.005)")
    args = parser.parse_args()

    setup_environment()

    # Define supported coins
    ALL_COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]
    
    if args.coins.lower() == "all":
        target_coins = ALL_COINS
    else:
        target_coins = [c.strip().lower() for c in args.coins.split(',')]

    # Configuration: Mature assets (BTC, ETH) need lower thresholds than volatile ones (SOL, DOGE).
    COIN_THRESHOLDS = {
        "bitcoin": 0.001,
        "ethereum": 0.001,
        "solana": 0.001,
        "dogecoin": 0.001
    }

    # Run Loop
    report = {}
    for coin in target_coins:
        if coin not in ALL_COINS:
            print(f"[WARN] Skipping unsupported coin: {coin}")
            continue
        
        # Use specific threshold if available, else default to args.threshold
        threshold = COIN_THRESHOLDS.get(coin, args.threshold)
            
        result = process_coin(coin, args.days, threshold)
        if result:
            report[coin] = result

    # Final Summary
    print("\n" + "="*50)
    print("FINAL PORTFOLIO REPORT")
    print("="*50)
    print(f"{'COIN':<10} | {'STRATEGY':<10} | {'RETURN':<8} | {'SHARPE':<6} | {'DD':<8}")
    print("-" * 65)
    for coin, data in report.items():

        print(f"{data['coin'].capitalize():<10} | Random Forest| {data['rf_metrics']['Total Return']:<8} | {data['rf_metrics']['Sharpe Ratio']:<6} | {data['rf_metrics']['Max Drawdown']:<8}")
        print(f"{'':<10} | Deep Learning| {data['dl_metrics']['Total Return']:<8} | {data['dl_metrics']['Sharpe Ratio']:<6} | {data['dl_metrics']['Max Drawdown']:<8}")
        print(f"{'':<10} | Buy&Hold     | {data['bh_metrics']['Total Return']:<8} | {data['bh_metrics']['Sharpe Ratio']:<6} | {data['bh_metrics']['Max Drawdown']:<8}")
        print(f"{'':<10} | SMA Cross    | {data['sma_metrics']['Total Return']:<8} | {data['sma_metrics']['Sharpe Ratio']:<6} | {data['sma_metrics']['Max Drawdown']:<8}")
        print("-" * 65)
    print("="*50)

if __name__ == "__main__":
    main()
