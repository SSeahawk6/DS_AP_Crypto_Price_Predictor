import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
        Visualizes Strategy Value vs. Asset Price for the report.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.history, label='ML Strategy Value', color='blue')
        plt.title('Strategy Performance')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(df['price'].values, label='BTC Price (Buy & Hold)', color='gray', alpha=0.5)
        plt.title('Asset Price Context')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        