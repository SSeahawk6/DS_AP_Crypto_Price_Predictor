# Predictive ML Model for Crypto Price Movements

An end-to-end Machine Learning pipeline that predicts future cryptocurrency price movements. This project moves beyond static "rule-based" trading by using a **Random Forest Classifier** to learn from historical technical indicators and predict whether a price will rise or fall.

---

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installation](#-installation)
  - [Reproducibility](#-reproducibility)
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
* **yfinance** (API Data Collection)

## ✨ Key Features

* **Real-World Data Ingestion**: Automatically fetches historical price data (OHLC) from **Yahoo Finance**.

* **Advanced Feature Engineering**: Calculates technical indicators like RSI, Moving Averages, and Bollinger Bands to create a feature matrix for the model.
* **Predictive Modeling**: Trains a **Random Forest Classifier** to predict if the price will be higher in $N$ hours.
* **Strict Evaluation**: Uses a time-series split (Train on Past, Test on Future) to prevent "look-ahead bias."
* **Backtesting Engine**: Simulates trading based on the model's predictions and calculates Profit & Loss (P&L) compared to a baseline.

---

# 📂 Project Structure

The project is organized as a modular Python application:

```text
crypto-ml-project/
├── data/raw/               # Stores downloaded CSVs (to avoid re-fetching)
├── notebooks/              # Jupyter notebooks for interactive demos
├── results/                # Stores generated reports and charts
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Handles Yahoo Finance data fetching and caching
│   ├── feature_engineer.py # Calculates technical indicators (RSI, BB, etc.)
│   ├── models.py           # Defines Random Forest and Deep Learning models
│   ├── evaluation.py       # Handles backtesting, financial metrics, and plotting
│   └── utils.py            # Helper functions for setup and reproducibility
├── tests/                  # Unit and integration tests (pytest)
├── main.py                 # Main CLI entry point (supports multi-coin loop)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
