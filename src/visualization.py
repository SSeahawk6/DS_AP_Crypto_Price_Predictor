import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

"""
------------------------------------------------------------------------
Visualization Module
------------------------------------------------------------------------
This module handles all graphical outputs for the Cryptocurrency 
Volatility Prediction project. It produces publication-quality figures 
for the technical report and evaluates model performance.
"""

def set_style():
    """Sets a professional plotting style for the report."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')
    
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
