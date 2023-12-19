#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
tickers = ["V3AL.L","EUFN","VWO","VT","IEUR","EEM","PPH","TAN","DGTL.L","HDRO"]
end_date = datetime.today()
start_date = end_date - timedelta(days = 5*365)
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start = start_date,end = end_date)
    adj_close_df[ticker] = data['Adj Close']
monthly_returns =  adj_close_df/adj_close_df.shift(1)-1
monthly_returns= monthly_returns.dropna()
cov_matrix = monthly_returns.cov()*12
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, monthly_returns):
    return np.sum(monthly_returns.mean()*weights)*12

def sharpe_ratio(weights, monthly_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, monthly_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
           
from fredapi import Fred

fred= Fred(api_key= "afa34cd62e7df712a63918bd9fc4555f")
ten_year_treasury_rate= fred.get_series_latest_release('GS10')/ 100

risk_free_rate = ten_year_treasury_rate.iloc[-1]
print(f"Risk Free Rate: {risk_free_rate:.4f}")


def neg_sharpe_ratio(weights, monthly_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, monthly_returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(tickers))]
initial_weights = np.array([1/len(tickers)]*len(tickers))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(monthly_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights = optimized_results.x

print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

optimal_portfolio_return = expected_return(optimal_weights, monthly_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, monthly_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")




