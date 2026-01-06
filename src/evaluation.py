# This module handles backtesting and result visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# Visualization Tools
# -------------------------

def set_style():
    """Sets a professional plotting style for the report."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')
    
    # We use a consistent style across all plots to ensure the final report looks professional.
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['font.size'] = 12

def plot_price_history(df, coin_name):
    """Plots raw price history to visualize market cycles."""
    set_style()
    plt.figure()
    plt.plot(df.index, df['price'], label='Close Price', color='blue', alpha=0.7)
    plt.title(f'{coin_name.capitalize()} Price History (Yahoo Finance)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    save_path = f'results/{coin_name}_price_history.png'
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved price chart to {save_path}")

def plot_technical_indicators(df, coin_name):
    """Visualizes calculated indicators like RSI and Bollinger Bands."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Bollinger Bands
    ax1.plot(df.index, df['price'], label='Price', color='black', alpha=0.5)
    ax1.plot(df.index, df['bb_upper'], label='BB Upper', color='green', linestyle='--')
    ax1.plot(df.index, df['bb_lower'], label='BB Lower', color='red', linestyle='--')
    ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='gray', alpha=0.1)
    ax1.set_title(f'{coin_name.capitalize()} - Bollinger Bands')
    ax1.legend()
    
    # RSI
    ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    plt.tight_layout()
    save_path = f'results/{coin_name}_indicators.png'
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, model_name):
    """Generates a heatmap of model predictions vs actual outcomes."""
    set_style()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    save_path = f'results/confusion_matrix_{model_name}.png'
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names):
    """Visualizes which indicators had the most predictive power."""
    if not hasattr(model, 'feature_importances_'):
        return
    set_style()
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()


# -------------------------
# Backtester
# -------------------------

class Backtester:
    """
    A class to simulate trading strategies and calculate Professional Financial Metrics.
    
    Attributes:
        initial_balance (float): Starting capital.
        cash (float): Current liquid cash available for trades.
        holdings (float): Current amount of crypto asset held.
        history (list): Time-series of total portfolio value.
    """
    
    def __init__(self, initial_balance=10000):
        """
        Initializes the backtester with a starting balance.
        Args:
            initial_balance (float): The starting capital (default 10,000).
        """
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.holdings = 0
        self.history = []

    def buy(self, price, date=None):
        """
        Executes a buy order using all available cash.
        """
        if self.cash > 0:
            self.holdings = self.cash / price
            self.cash = 0

    def sell(self, price, date=None):
        """
        Executes a sell order for all current holdings.
        """
        if self.holdings > 0:
            self.cash = self.holdings * price
            self.holdings = 0

    def run(self, df):
        """
        Runs the simulation using a DataFrame containing 'price' and 'prediction'.
        Args:
            df (pd.DataFrame): Data with columns 'price' and 'prediction'.
        Returns:
            pd.DataFrame: Portfolio history for visualization.
        """
        self.cash = self.initial_balance
        self.holdings = 0
        self.history = []
        
        prices = df['price'].values
        predictions = df['prediction'].values
        
        for i in range(len(predictions)):
            current_price = prices[i]
            prediction = predictions[i]
            
            # Prediction 1 = Buy; Prediction 0 = Sell
            if prediction == 1:
                self.buy(current_price)
            elif prediction == 0:
                self.sell(current_price)
                
            # Track total portfolio value (Cash + Asset Value)
            current_value = self.cash + (self.holdings * current_price)
            self.history.append(current_value)
            
        return pd.DataFrame({'value': self.history}, index=df.index)

    def calculate_metrics(self):
        """
        Calculates Sharpe Ratio and Max Drawdown based on trade history.
        Returns:
            dict: Financial performance metrics.
        """
        if not self.history:
            return {
                "Total Return": "0.00%",
                "Sharpe Ratio": "0.00",
                "Max Drawdown": "0.00%"
            }

        portfolio_df = pd.DataFrame(self.history, columns=['value'])
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        # --- FIX APPLIED HERE ---
        # The last value in history represents our final Net Worth (Cash + Crypto)
        final_value = self.history[-1]
        
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        mean_return = portfolio_df['returns'].mean()
        std_return = portfolio_df['returns'].std()
        
        # Annualized Sharpe Ratio calculation (assuming Daily data)
        # We assume 365 trading days for crypto (unlike 252 for stocks) because crypto trades 24/7.
        sharpe = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0.0
            
        # Max Drawdown calculation
        rolling_max = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }

    def plot_results(self, df, filename="results/backtest_chart.png"):
        """
        Visualizes Strategy Value, Buy/Sell Signals, and Drawdown Analysis.
        Generates a 3-panel professional financial report chart.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        set_style()
        
        # Create 3 subplots sharing the x-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
        
        # --- Panel 1: Strategy Performance ---
        ax1.plot(df.index, self.history, label='ML Strategy Value ($)', color='#1f77b4', linewidth=2)
        ax1.set_title('Strategy Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        
        # --- Panel 2: Price Action & Trade Signals ---
        ax2.plot(df.index, df['price'], label='Asset Price (Hold)', color='gray', alpha=0.5)
        
        # Identify Buy/Sell points
        # Buy when prediction becomes 1 (and wasn't before) - simplified logic: buy signal is whenever pred=1
        # For visualization, we plot a marker every time we HOLD the asset? 
        # Better: Plot marker only on CHANGE of position.
        
        buy_signals = df[df['prediction'] == 1]
        sell_signals = df[df['prediction'] == 0]
        
        # We only want to plot markers where the signal CHANGED to avoid clutter
        # Create a 'signal_change' mask
        df['signal_change'] = df['prediction'].diff()
        
        buys = df[df['signal_change'] == 1]  # 0 -> 1
        sells = df[df['signal_change'] == -1] # 1 -> 0
        
        ax2.scatter(buys.index, buys['price'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
        ax2.scatter(sells.index, sells['price'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)
        
        ax2.set_title('Trade Signals (Green=Buy, Red=Sell)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price ($)')
        ax2.legend(loc='upper left')

        # --- Panel 3: Drawdown Analysis ---
        # Calculate Drawdown series
        portfolio_series = pd.Series(self.history, index=df.index)
        rolling_max = portfolio_series.cummax()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        
        ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        ax3.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Risk Analysis (Drawdown)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('% from Peak')
        ax3.set_xlabel('Date')
        
        # Format percentages on Y-axis for drawdown
        vals = ax3.get_yticks()
        ax3.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300) # High Resolution for Report
        plt.close()
        print(f"[INFO] Saved high-res chart to {filename}")
