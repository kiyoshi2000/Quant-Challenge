import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from outils import *
from backtest import *

# Main execution block
if __name__ == "__main__":
    # Define parameters
    tickers = ['NVDA', 'BRK-B', 'C', 'JPM', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'JNJ']
    start_date = '2020-01-01'
    end_date = '2024-04-30'
    initial_investment = 100000

    # Download historical adjusted closing prices
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Remove timezone info
    data.index = data.index.tz_localize(None)

    # Download benchmark data
    benchmark_ticker = '^GSPC'
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
    benchmark_data.index = benchmark_data.index.tz_localize(None)

    # Run backtest with PSO optimization
    portfolio_returns, portfolio_allocations, cumulative_returns = backtest_pso(
        data, tickers, start_date, end_date, rebalance_period='3ME', initial_investment=initial_investment
    )

    # Calculate benchmark cumulative returns
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() * initial_investment

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns['Portfolio Value'], label='Optimized Portfolio')
    plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns.values, label='S&P 500 Benchmark')
    plt.title('Cumulative Returns: Portfolio vs. S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate daily returns
    portfolio_values = cumulative_returns['Portfolio Value']
    portfolio_daily_returns = portfolio_values.pct_change().dropna()
    benchmark_daily_returns = benchmark_returns

    # Calculate risk metrics for the portfolio
    portfolio_annual_return = calculate_annualized_return(portfolio_daily_returns)
    portfolio_annual_volatility = calculate_annualized_volatility(portfolio_daily_returns)
    portfolio_sharpe_ratio = calculate_sharpe_ratio(portfolio_daily_returns)
    portfolio_max_drawdown = calculate_max_drawdown(portfolio_values)

    # Calculate risk metrics for the benchmark
    benchmark_annual_return = calculate_annualized_return(benchmark_daily_returns)
    benchmark_annual_volatility = calculate_annualized_volatility(benchmark_daily_returns)
    benchmark_sharpe_ratio = calculate_sharpe_ratio(benchmark_daily_returns)
    benchmark_max_drawdown = calculate_max_drawdown(benchmark_cumulative_returns)

    # Print risk metrics
    print("Risk Metrics:")
    print("\nOptimized Portfolio:")
    print(f"Annualized Return: {portfolio_annual_return:.2%}")
    print(f"Annualized Volatility: {portfolio_annual_volatility:.2%}")
    print(f"Sharpe Ratio: {portfolio_sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {portfolio_max_drawdown:.2%}")

    print("\nS&P 500 Benchmark:")
    print(f"Annualized Return: {benchmark_annual_return:.2%}")
    print(f"Annualized Volatility: {benchmark_annual_volatility:.2%}")
    print(f"Sharpe Ratio: {benchmark_sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {benchmark_max_drawdown:.2%}")