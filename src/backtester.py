import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    """
    A class to simulate trading strategies and calculate Professional Financial Metrics.
    """
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.history = []

    def run(self, prices, predictions):
        """
        Executes the backtest.
        """
        self.balance = self.initial_balance
        self.position = 0
        self.history = []
        
        for i in range(len(predictions) - 1):
            current_price = prices.iloc[i]
            prediction = predictions[i]
            
            # Buy if prediction is 1 and we have cash
            if prediction == 1 and self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0
                
            # Sell if prediction is 0 and we hold coin
            elif prediction == 0 and self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
                
            # Track total portfolio value
            current_value = self.balance + (self.position * current_price)
            self.history.append(current_value)
            
        # Final Sell
        final_price = prices.iloc[-1]
        if self.position > 0:
            self.balance = self.position * final_price
            self.position = 0
            
        self.history.append(self.balance)
        return self.balance

    def calculate_metrics(self):
        """
        Calculates Sharpe Ratio and Max Drawdown.
        """
        portfolio_df = pd.DataFrame(self.history, columns=['value'])
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        # 1. Total Return
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        # 2. Sharpe Ratio (Risk-Adjusted Return)
        # We assume 252 trading days in a year for annualization
        mean_return = portfolio_df['returns'].mean()
        std_return = portfolio_df['returns'].std()
        
        if std_return == 0 or np.isnan(std_return):
            sharpe = 0.0
        else:
            sharpe = (mean_return / std_return) * np.sqrt(365)
            
        # 3. Max Drawdown (Worst drop from peak)
        rolling_max = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }

    def plot_results(self, prices, filename="results/backtest_chart.png"):
        """
        Visualizes the portfolio value vs the Asset Price.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Portfolio Value
        plt.subplot(2, 1, 1)
        plt.plot(self.history, label='Strategy Value', color='blue')
        plt.title('Portfolio Performance')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Asset Price (Buy & Hold comparison)
        plt.subplot(2, 1, 2)
        plt.plot(prices.values, label='Asset Price (Buy & Hold)', color='gray', alpha=0.5)
        plt.title('Underlying Asset Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"[INFO] Chart saved to {filename}")
        plt.close()
