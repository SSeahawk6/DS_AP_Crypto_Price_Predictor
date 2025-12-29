import pandas as pd
import matplotlib.pyplot as plt

class Backtester:
    """
    A class to simulate trading strategies and calculate Profit & Loss (P&L).
    """
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0 = No position, 1 = Holding the coin
        self.history = []

    def run(self, prices, predictions):
        """
        Executes the backtest simulation on historical data.

        Iterates through the price data and executes buy/sell orders based on
        the provided ML predictions.

        Args:
            prices (pd.Series): A pandas Series containing actual price data.
            predictions (np.array): A numpy array of binary predictions (1=Up, 0=Down).

        Returns:
            float: The final portfolio balance in USD.
        """
        # Reset balance for a new run
        self.balance = self.initial_balance
        self.position = 0
        self.history = []
        
        # Loop through each day in the test set
        for i in range(len(predictions) - 1):
            current_price = prices.iloc[i]
            prediction = predictions[i]
            
            # Simple Strategy:
            # If model predicts UP (1) and we have cash -> BUY
            if prediction == 1 and self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0
                
            # If model predicts DOWN (0) and we hold coin -> SELL
            elif prediction == 0 and self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
                
            # Record total value (Cash + Coin Value)
            current_value = self.balance + (self.position * current_price)
            self.history.append(current_value)
            
        # Sell everything at the end to finalize P&L
        final_price = prices.iloc[-1]
        if self.position > 0:
            self.balance = self.position * final_price
            self.position = 0
            
        self.history.append(self.balance)
        return self.balance

    def plot_results(self, prices, filename="results/backtest_chart.png"):
        """
        Visualizes the portfolio value over time.
        Args:
            prices: The price data series.
            filename (str): Path to save the chart image.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, label='ML Strategy Value')
        plt.title('Backtest Results: ML Strategy vs Time')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save to the specific unique filename provided
        plt.savefig(filename)
        print(f"[INFO] Chart saved to {filename}")
        plt.close()  # Important: Close plot to free memory in parallel loops
