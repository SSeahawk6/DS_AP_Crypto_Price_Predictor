# Predictive ML Model for Crypto Price Movements

An end-to-end Machine Learning pipeline that predicts future cryptocurrency price movements. This project moves beyond static "rule-based" trading by using a **Random Forest Classifier** to learn from historical technical indicators and predict whether a price will rise or fall.

---

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contact](#contact)

---

## 🤖 About the Project

Cryptocurrency markets are highly volatile, and traditional algorithmic trading strategies often rely on rigid, pre-defined rules (e.g., "Buy if RSI < 30"). These static strategies fail to adapt to changing market conditions.

This project solves that problem by implementing a **Machine Learning approach**. It fetches real-world historical data, engineers complex technical features (RSI, Bollinger Bands, MACD), and trains a `scikit-learn` Random Forest model to predict price direction. The model's performance is then rigorously evaluated against a "Buy & Hold" strategy using a custom backtesting engine.

### Built With
* **Python 3.10+**
* **scikit-learn** (Machine Learning)
* **pandas & NumPy** (Data Processing)
* **matplotlib** (Visualization)
* **requests** (API Data Collection)

## ✨ Key Features

* **Real-World Data Ingestion**: Automatically fetches historical price data (OHLC) from the **CoinGecko API**.
* **Advanced Feature Engineering**: Calculates technical indicators like RSI, Moving Averages, and Bollinger Bands to create a feature matrix for the model.
* **Predictive Modeling**: Trains a **Random Forest Classifier** to predict if the price will be higher in $N$ hours.
* **Strict Evaluation**: Uses a time-series split (Train on Past, Test on Future) to prevent "look-ahead bias."
* **Backtesting Engine**: Simulates trading based on the model's predictions and calculates Profit & Loss (P&L) compared to a baseline.

---

# 📂 Project Structure

The project is organized as a modular Python application:

```text
crypto-ml-project/
├── data/                   # Stores downloaded CSVs (to avoid re-fetching)
├── src/
│   ├── __init__.py
│   ├── api_collector.py    # Handles API requests and rate limits
│   ├── feature_engineer.py # Calculates technical indicators (RSI, MACD, etc.)
│   ├── ml_model.py         # Trains the Random Forest and generates predictions
│   └── backtester.py       # Simulates trading and calculates P&L
├── tests/                  # Unit and integration tests (pytest)
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
