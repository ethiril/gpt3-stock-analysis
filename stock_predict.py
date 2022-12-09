# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from yfinance import Tickers

def mean_reversion(data):
    """Calculate the mean reversion for the given stock data"""
    return data['Close'].mean()

def vwap(data):
    """Calculate the volume weighted average price (VWAP) for the given stock data"""
    return data['Adj Close'].multiply(data['Volume']).sum() / data['Volume'].sum()

def twap(data):
    """Calculate the time weighted average price (TWAP) for the given stock data"""
    return data['Adj Close'].sum() / len(data)

def pov(data):
    """Calculate the percentage of volume for the given stock data"""
    return data['Volume'].pct_change()

def daily_returns(data):
    """Calculate the daily returns for the given stock data"""
    return data['Close'].pct_change()

def sharpe_ratio(data, risk_free_rate=0.01):
    """Calculate the Sharpe ratio for the given stock data"""
    daily_returns = data['Close'].pct_change()
    excess_return = daily_returns - risk_free_rate
    return excess_return.mean() / excess_return.std()

def macd(data, short_window=12, long_window=26):
    """Calculate the moving average convergence divergence (MACD) for the given stock data"""
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    return ema_short - ema_long

def rsi(data, periods=14):
    """Calculate the relative strength index (RSI) for the given stock data"""
    # Calculate the gain and loss for each period
    gain = data['Close'].diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss *= -1
    
    # Calculate the average gain and loss
    avg_gain = gain.rolling(periods).mean()
    avg_loss = loss.rolling(periods).mean()
    
    # Calculate the relative strength
    relative_strength = avg_gain / avg_loss
    
    # Calculate the RSI
    return 100 - (100 / (1 + relative_strength))

top_tickers = pd.read_html('https://finance.yahoo.com/gainers')

# Remove invalid ticker symbols from the list
ticker_list = [ticker for ticker in ticker_list if ticker.isalnum()]

# Join the list of ticker symbols into a string
ticker_string = ' '.join(ticker_list)

# Get the stock data from Yahoo Finance
stock_data = yf.download(ticker_string, period='1y')



# Calculate the financial metrics for each stock
metrics = {}
for ticker in ticker_list:
    metrics[ticker] = {
        "mean_reversion": mean_reversion(stock_data[ticker]),
        "vwap": vwap(stock_data[ticker]),
        "twap": twap(stock_data[ticker]),
        "pov": pov(stock_data[ticker]),
        "daily_returns": daily_returns(stock_data[ticker]),
        "sharpe_ratio": sharpe_ratio(stock_data[ticker]),
        "macd": macd(stock_data[ticker]),
        "rsi": rsi(stock_data[ticker])
        }


# Print the financial metrics for each stock
for ticker, values in metrics.items():
    print(f"Metrics for {ticker}:")
for metric, value in values.items():
    print(f"{metric}: {value}")
print()
