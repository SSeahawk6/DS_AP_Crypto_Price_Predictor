import argparse
import pandas as pd
from src.data_loader import fetch_crypto_data
from src.feature_engineer import add_technical_indicators
from src.models import train_rf_model, ModelOptimizer, train_dl_model
from src.evaluation import Backtester, plot_feature_importance
from src.utils import setup_environment

def process_coin(coin, days, initial_balance=10000):
    """
    Runs the full pipeline for a single coin.
    Returns a dictionary of metrics.
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING: {coin.upper()}")
    print(f"{'='*50}")
    
    # 1. Data Collection
    csv_path = fetch_crypto_data(coin, days)
    if not csv_path:
        return None

    # 2. Feature Engineering
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_features = add_technical_indicators(df)
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
    # We pass the FULL df_features, the function handles the split consistently
    dl_model, _, _, dl_preds = train_dl_model(df_features)

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
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, default="all", 
                        help="Comma-separated list of coins or 'all' (default: all)")
    parser.add_argument("--days", type=int, default=1825, help="Number of days of data (default: 5 years)")
    args = parser.parse_args()

    setup_environment()

    # Define supported coins
    ALL_COINS = ["bitcoin", "ethereum", "solana", "dogecoin", "cardano"]
    
    if args.coins.lower() == "all":
        target_coins = ALL_COINS
    else:
        target_coins = [c.strip().lower() for c in args.coins.split(',')]

    # Run Loop
    report = {}
    for coin in target_coins:
        if coin not in ALL_COINS:
            print(f"[WARN] Skipping unsupported coin: {coin}")
            continue
            
        metrics = process_coin(coin, args.days)
        if metrics:
            report[coin] = metrics

    # Final Summary
    print("\n" + "="*50)
    print("FINAL PORTFOLIO REPORT")
    print("="*50)
    print(f"{'COIN':<15} | {'RETURN':<10} | {'SHARPE':<10} | {'DRAWDOWN':<10}")
    print("-" * 55)
    for coin, m in report.items():
        print(f"{coin.capitalize():<15} | {m['Total Return']:<10} | {m['Sharpe Ratio']:<10} | {m['Max Drawdown']:<10}")
    print("="*50)

if __name__ == "__main__":
    main()
