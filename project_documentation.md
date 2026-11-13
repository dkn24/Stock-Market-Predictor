# Stock Market Price Prediction Model

A Python-based financial forecasting tool that downloads historical stock data, engineers technical indicators, and uses linear models (Linear Regression and Lasso) to predict future closing prices.

## Overview

This project fetches 16 years of historical stock data via Yahoo Finance (`yfinance`), calculates short- and medium-term moving averages, and trains a regression model to forecast closing prices **5 trading days into the future**. Results are evaluated using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE), then visualized on an interactive dark-mode chart using Plotly.

> **Note:** The current script implements `LinearRegression`. You can easily swap in or compare `Lasso` from `sklearn.linear_model` to introduce $L_1$ regularization and prevent overfitting when adding more technical indicators.

---

## Features

- **Automated Data Ingestion:** Downloads historical OHLCV data for any user-specified stock ticker (e.g., `AAPL`, `MSFT`, `GOOGL`) from `2010-01-01` to `2026-11-13`.
- **Feature Engineering:** Computes 5-day (`MA_5`) and 20-day (`MA_20`) simple moving averages to capture short- and medium-term price momentum.
- **Time-Series Aware Splitting:** Uses chronological splitting (`shuffle=False`) to ensure the model trains strictly on past data (80%) and tests on unseen future data (20%), preventing data leakage.
- **Performance Metrics:** Calculates both dollar-value error (**MAE**) and relative error (**Percentage Error**).
- **Interactive Visualization:** Generates an interactive Plotly line chart comparing actual vs. predicted closing prices.

---

## Prerequisites & Installation

1. **Clone or download** the project files into a local directory.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the main Python script from your terminal:

```bash
python main.py
```

When prompted, enter a valid stock ticker symbol:
```text
Enter the stock ticker (e.g., AAPL, MSFT): AAPL
```

The script will output the training/testing split sizes, a table of the most recent predictions alongside their percentage errors, overall accuracy metrics, and automatically launch the interactive Plotly graph in your default web browser.

---

## Extending with Lasso Regression

Since you plan to use **Lasso Regression** alongside Linear Regression, you can import `Lasso` from `scikit-learn` (`from sklearn.linear_model import LinearRegression, Lasso`) to penalize less important features—especially useful if you expand your feature set with RSI, MACD, or volume indicators:

```python
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)
lasso_predictions = lasso_model.predict(x_test)
```

---

## Disclaimer

*This project is for educational and analytical purposes only. Stock market prices are inherently volatile and unpredictable; do not use these predictions as financial advice or for live trading decisions.*