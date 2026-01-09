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
    
    # 1. Data Collection
    csv_path = fetch_crypto_data(coin, days)
    if not csv_path:
        return None

    # 2. Feature Engineering
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_features = add_technical_indicators(df, threshold=threshold)
    if df_features.empty:
        print("[ERROR] No data after processing.")
        return None

    # 3. Prepare Data
    X = df_features.drop(['target', 'next_day_return', 'price'], axis=1)
    y = df_features['target']
    
    # Time-based split for training (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ---------------------------------------------------------
    # STRETCH GOAL 1: Model Comparison (Random Forest vs Deep Learning)
    # ---------------------------------------------------------
    
    # --- Model A: Random Forest (Optimized) ---
    print("\n[MODEL A] Random Forest Optimization...")
    optimizer = ModelOptimizer(n_splits=3)
    rf_model = optimizer.optimize_random_forest(X, y, n_iter=5)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # STRETCH GOAL 2: Feature Importance
    print("[INFO] Generating Feature Importance Chart...")
    plot_feature_importance(rf_model, X.columns, filename=f"results/feature_importance_{coin}.png")

    # --- Model B: Deep Learning (Neural Network) ---
    print("\n[MODEL B] Deep Learning Training...")
    # train_dl_model handles scaling internally, passing raw df is safer
    try:
        dl_model, _, _, dl_preds = train_dl_model(df_features)
    except Exception as e:
        print(f"[WARN] DL Model Failed: {e}")
        dl_preds = []

    # ---------------------------------------------------------
    # Backtesting (Using Best Model - let's default to Random Forest for consistency)
    # ---------------------------------------------------------
    print("\n--- Starting Backtest (Random Forest Strategy) ---")
    
    # We need to align predictions with the test dataframe index
    test_df = df_features.iloc[split:].copy()
    
    # Ensure lengths match (sometimes DL drops different number of rows, but here we use RF)
    if len(rf_preds) != len(test_df):
        print(f"[WARNING] Length mismatch: Preds {len(rf_preds)} vs Test DF {len(test_df)}")
        test_df = test_df.iloc[-len(rf_preds):]
        
    test_df['prediction'] = rf_preds
    
    backtester = Backtester(initial_balance=initial_balance)
    backtester.run(test_df)
    metrics = backtester.calculate_metrics()
    
    print(f"[RESULT] {coin.capitalize()} Metrics: {metrics}")
    backtester.plot_results(test_df, filename=f"results/chart_{coin}.png")

    # --- Benchmark: Buy & Hold ---
    print("--- Calculating Benchmark (Buy & Hold) ---")
    test_df_bh = test_df.copy()
    test_df_bh['prediction'] = 1  # Always Buy
    
    bh_backtester = Backtester(initial_balance=initial_balance)
    bh_backtester.run(test_df_bh)
    bh_metrics = bh_backtester.calculate_metrics()
    print(f"[BENCHMARK] {coin.capitalize()} B&H Metrics: {bh_metrics}")

    # --- Benchmark: SMA Crossover (Rule-Based) ---
    print("--- Calculating Benchmark (SMA Crossover) ---")
    test_df_sma = test_df.copy()
    # Signal: 1 if SMA_20 > SMA_50 (Golden Cross), else 0
    test_df_sma['prediction'] = (test_df_sma['sma_20'] > test_df_sma['sma_50']).astype(int)
    
    sma_backtester = Backtester(initial_balance=initial_balance)
    sma_backtester.run(test_df_sma)
    sma_metrics = sma_backtester.calculate_metrics()
    print(f"[BENCHMARK] {coin.capitalize()} SMA Cross Metrics: {sma_metrics}")

    # --- Generate Comparison Chart ---
    strategies = {
        'ML Strategy': backtester.history,
        'Buy & Hold': bh_backtester.history,
        'SMA Crossover': sma_backtester.history
    }
    plot_strategy_comparison(test_df, strategies, filename=f"results/comparison_{coin}.png")
    
    return metrics, bh_metrics, sma_metrics

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

    # Dynamic Threshold Configuration
    # Rationale: Mature assets (BTC, ETH) need lower thresholds to capture steady growth.
    # Volatile assets (SOL, DOGE) need higher thresholds to filter noise.
    COIN_THRESHOLDS = {
        "bitcoin": 0.001,
        "ethereum": 0.001,
        "solana": 0.005,
        "dogecoin": 0.005
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
            metrics, bh_metrics, sma_metrics = result
            report[coin] = {'ml': metrics, 'bh': bh_metrics, 'sma': sma_metrics}

    # Final Summary
    print("\n" + "="*50)
    print("FINAL PORTFOLIO REPORT")
    print("="*50)
    print(f"{'COIN':<10} | {'STRATEGY':<10} | {'RETURN':<8} | {'SHARPE':<6} | {'DD':<8}")
    print("-" * 65)
    for coin, data in report.items():
        ml = data['ml']
        bh = data['bh']
        sma = data['sma']
        print(f"{coin.capitalize():<10} | {'ML':<10} | {ml['Total Return']:<8} | {ml['Sharpe Ratio']:<6} | {ml['Max Drawdown']:<8}")
        print(f"{'':<10} | {'Buy&Hold':<10} | {bh['Total Return']:<8} | {bh['Sharpe Ratio']:<6} | {bh['Max Drawdown']:<8}")
        print(f"{'':<10} | {'SMA Cross':<10} | {sma['Total Return']:<8} | {sma['Sharpe Ratio']:<6} | {sma['Max Drawdown']:<8}")
        print("-" * 65)
    print("="*50)

if __name__ == "__main__":
    main()
