import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
# from outils import *
from backtest import *
from metrics import PerformanceMetrics

# TODO: adicionar stress testing em um períodod conturbado
# TODO: adicionar a seleção de ações (agora as ações são sempre as mesmas, precisa escolher as ações a cada janela de tempo para poder fazer o backtesting e stress testing)
# ? melhor usar algo genético?

# Main execution block
if __name__ == "__main__":
    # Define parameters
    tickers = ['NVDA', 'BRK-B', 'C', 'JPM', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'JNJ',
           'TLT',  # Long-term Treasury ETF
           'GLD',  # Gold ETF
           'EFA',  # MSCI EAFE ETF (International stocks)
           'EEM',  # Emerging Markets ETF
           'LQD']  # Investment Grade Corporate Bond ETF
    
    start_date = '2015-01-01'
    end_date = '2024-04-30'
    initial_investment = 1_000_000

    # Download historical adjusted closing prices
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Remove timezone info
    data.index = data.index.tz_localize(None)

    # Download benchmark data
    benchmark_ticker = '^GSPC'
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
    benchmark_data.index = benchmark_data.index.tz_localize(None)

    # Run backtest with PSO optimization
    portfolio_values, allocations_df = backtest_pso(
        data, tickers, start_date, end_date, rebalance_period='1ME', initial_investment=initial_investment
    )

    # Ensure allocations_df index is datetime and sorted
    allocations_df.index = pd.to_datetime(allocations_df.index)
    allocations_df = allocations_df.sort_index()

    # Extend allocations to daily frequency by forward-filling
    allocations_daily = allocations_df.reindex(data.index, method='ffill')

    # Remove any NaN values
    portfolio_values = portfolio_values.dropna()

    # Calculate benchmark cumulative returns
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() * initial_investment

    # Align benchmark data with portfolio dates
    benchmark_cumulative_returns = benchmark_cumulative_returns[benchmark_cumulative_returns.index.isin(portfolio_values.index)]

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values.values, label='Optimized Portfolio')
    plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns.values, label='S&P 500 Benchmark')
    plt.title('Cumulative Returns: Portfolio vs. S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the allocations
    plt.figure(figsize=(12, 6))
    allocations_daily.plot.area(stacked=True, ax=plt.gca())  # Define o eixo atual para evitar conflito
    plt.title('Portfolio Allocations Over Time')
    plt.xlabel('Date')
    plt.ylabel('Allocation Percentage')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()

    # Calculate daily returns
    portfolio_daily_returns = portfolio_values.pct_change().dropna()
    benchmark_daily_returns = benchmark_returns[benchmark_returns.index.isin(portfolio_daily_returns.index)]

    metrics = PerformanceMetrics()

    # Calculate risk metrics for the portfolio
    portfolio_annual_return = metrics.calculate_annualized_return(portfolio_daily_returns)
    portfolio_annual_volatility = metrics.calculate_annualized_volatility(portfolio_daily_returns)
    portfolio_sharpe_ratio = metrics.calculate_sharpe_ratio(portfolio_daily_returns)
    portfolio_max_drawdown = metrics.calculate_max_drawdown(portfolio_values)

    # Calculate risk metrics for the benchmark
    benchmark_annual_return = metrics.calculate_annualized_return(benchmark_daily_returns)
    benchmark_annual_volatility = metrics.calculate_annualized_volatility(benchmark_daily_returns)
    benchmark_sharpe_ratio = metrics.calculate_sharpe_ratio(benchmark_daily_returns)
    benchmark_max_drawdown = metrics.calculate_max_drawdown(benchmark_cumulative_returns)

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