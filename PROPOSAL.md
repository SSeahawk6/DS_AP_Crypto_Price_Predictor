# Project Proposal: Predictive ML Model for Crypto Price Movements

**Student:** Jean Trochet
**Course:** Advanced Programming - Fall 2025
**Category:** Machine Learning / Statistical Analysis Tools

## 1. Problem Statement & Motivation
Cryptocurrency price movements are highly volatile and difficult to predict. Traditional "algorithmic trading" often relies on static, pre-defined rules (e.g., "buy when RSI < 30") which fail to adapt to changing market conditions.

My motivation is to build an end-to-end **Machine Learning pipeline** to predict future price movements. The goal is to determine if a **Random Forest Classifier** trained on technical indicators can outperform a simple "Buy & Hold" strategy.

## 2. Planned Approach & Technologies
The project is a modular command-line application built with Python 3.10+.

### Core Technologies
* **Data Collection:** `requests` to fetch historical OHLC data from the CoinGecko API.
* **Data Processing:** `pandas` and `NumPy` for time-series manipulation.
* **Machine Learning:** `scikit-learn` to implement the Random Forest Classifier.
* **Visualization:** `matplotlib` to plot P&L curves and trade signals.
* **Testing:** `pytest` for unit and integration testing (>70% coverage).

### Architecture
1.  **`src/api_collector.py`**: Robust data fetching with error handling.
2.  **`src/feature_engineer.py`**: Calculates technical indicators (RSI, Bollinger Bands, SMA) to create the feature matrix (X).
3.  **`src/ml_model.py`**: Trains the model using a time-series split (Train on Past, Test on Future) to avoid look-ahead bias.
4.  **`src/backtester.py`**: A simulation engine that executes trades based on model predictions and tracks portfolio value.

## 3. Success Criteria
1.  Successfully fetch 1+ years of daily data for Bitcoin/Ethereum.
2.  Correctly calculate at least 4 technical features (verified by unit tests).
3.  Train a Random Forest model that achieves >50% accuracy on the test set.
4.  Generate a comparative P&L chart (ML Strategy vs. Buy & Hold).
5.  Maintain a clean repository structure with comprehensive documentation.

## 4. Maintenance & Future Work
The codebase is maintained using `git` with semantic commit messages. Code quality is enforced via `flake8` and `black`. Future improvements could include implementing Deep Learning models (LSTM) or optimizing hyperparameters using Grid Search.
