# Cryptocurrency Price Direction Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

**An intelligent, automated trading agent that adapts its strategy based on asset volatility.** 

This project explores the application of Machine Learning (Random Forest) to predict the directional movement of major cryptocurrencies. Unlike static rule-based systems (e.g., "Buy if RSI < 30"), our agent learns optimal patterns from 5 years of historical data.

The key innovation of this project is the **Hybrid Threshold Strategy**, which dynamically adjusts the "risk appetite" of the agent based on the asset's maturity.

---

## 🚀 Key Features

*   **End-to-End Pipeline with Caching:** Automated data ingestion from Yahoo Finance, featuring **intelligent local caching** (CSVs) to prevent rate limits and speed up experimentation. 
*   **Hybrid Threshold Logic:** 
    *   **Low Threshold (0.1%):** For mature assets like **Bitcoin** and **Ethereum**, capturing steady, small gains.
    *   **High Threshold (0.5%):** For volatile assets like **Solana** and **Dogecoin**, filtering out market noise.
*   **Professional Backtesting:** Simulates real-world trading with "Out-of-Sample" testing (train on past, test on future).
*   **Robust Architecture:** Modular Python design with Unit Tests (`pytest`) and CI/CD integration.

---

## 🎮 Demo

Experience the **Hybrid Threshold Strategy** in action. Run the following command to process all assets:

```bash
python main.py --coins all --days 1825
```

**What happens next?**
1.  **Auto-Config:** The system detects the asset type and assigns the optimal threshold:
    *   `PROCESSING: BITCOIN | Threshold=0.001` (Capture Trend)
    *   `PROCESSING: SOLANA  | Threshold=0.005` (Filter Noise)
2.  **Training:** A Random Forest model is trained on 4 years of data.
3.  **Backtest:** The strategy is tested on the final year (Out-of-Sample).
4.  **Results:** Performance charts are generated in `results/`.

### 🐍 Python API Usage

You can also import the modules directly in a Jupyter Notebook or script:

```python
from src.data_loader import fetch_crypto_data
from src.feature_engineer import add_technical_indicators
from src.models import train_rf_model
from src.evaluation import Backtester

# 1. Fetch Data
csv_path = fetch_crypto_data("bitcoin", days=365)
df = pd.read_csv(csv_path, parse_dates=True, index_col='timestamp')

# 2. Add Features (Hybrid Threshold)
df = add_technical_indicators(df, threshold=0.001)

# 3. Train Model
model, X_test, y_test, predictions = train_rf_model(df)

# 4. Backtest
backtester = Backtester(initial_balance=10000)
backtester.run(pd.DataFrame({'price': ..., 'prediction': predictions}))
```

---

## 📊 Performance Highlights

The agent was benchmarked against a naive "Buy & Hold" strategy and a standard "SMA Crossover" rule over a 5-year period (2021-2025).

> **Note:** The full experimental results, including Total Return, Sharpe Ratio, and Drawdown comparisons for all assets, are detailed in the **[Project Report](report.tex)**.

The backtesting engine generates professional equity curves and risk metrics, proving that the **Hybrid Strategy** successfully outperforms the baseline on major assets while significantly reducing drawdown on volatile ones.

---

## 📂 Project Structure

The repository follows a production-grade structure:

```text
crypto-ml-project/
├── main.py                 # CLI Entry Point (Run this!)
├── requirements.txt        # Dependencies
├── report.tex              # Final Project Report (LaTeX)
├── data/raw/               # Cached market data (CSVs)
├── results/                # Generated charts and metrics
├── src/                    # Source Code
│   ├── data_loader.py      # API Wrapper & Caching
│   ├── feature_engineer.py # Technical Analysis (RSI, BB, MACD)
│   ├── models.py           # Random Forest Implementation
│   ├── evaluation.py       # Backtesting Engine & Plotting
│   └── utils.py            # Environment Setup
└── tests/                  # Automated Tests
```

---

## 🛠️ Usage

### 1. Installation
Clones the repo and install dependencies:
```bash
git clone https://github.com/your-username/crypto-price-predictor.git
cd crypto-price-predictor
pip install -r requirements.txt
```

### 2. Run the Pipeline
Run the full backtest for all coins using the Hybrid Strategy:
```bash
python main.py --coins all --days 1825
```

### 3. Verification
To verify the system integrity, run the test suite:
```bash
pytest
```

---

## ⚖️ Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Cryptocurrency trading involves significant risk. The author is not responsible for any financial losses.
