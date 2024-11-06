import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from outils import *
from backtest import *


if __name__ == "__main__":
    # Definir os parâmetros
    tickers = ['NVDA', 'BRK-B', 'C', 'JPM', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'JNJ']
    start_date = '2020-01-01'
    end_date = '2024-04-30'
    population_size = 50
    investment_amount = 100000

    # Baixar os dados
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Executar o backtest com PSO e recalcular o portfólio a cada 3 meses
    portfolio_returns, portfolio_allocations = backtest_pso(data, tickers, start_date, end_date, rebalance_period='3ME')
    # portfolio_returns, portfolio_allocations = backtest_pso_dynamic(data, tickers, start_date, end_date, rebalance_period='3M')

    print("Retornos do Portfólio ao longo do tempo:")
    print(portfolio_returns)

    print("\nAlocações do Portfólio ao longo do tempo:")
    print(portfolio_allocations)

    # Plotar os retornos ao longo do tempo
    plt.plot(portfolio_returns['Date'], portfolio_returns['Portfolio Return'])
    plt.title('Retornos do Portfólio PSO ao longo do tempo')
    plt.xlabel('Data')
    plt.ylabel('Retorno')
    plt.show()

    # Chamando a função para plotar os pesos do portfólio ao longo do tempo
    plot_portfolio_allocations(portfolio_allocations, tickers)

    # 1. Baixar os dados do S&P 500
    benchmark_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']

    # 2. Calcular os retornos cumulativos
    # Retornos cumulativos do portfólio
    portfolio_cumulative_returns = (1 + portfolio_returns['Portfolio Return']).cumprod()

    # Retornos do benchmark e retornos cumulativos
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()

    # 3. Alinhar as datas
    combined_returns = pd.DataFrame({
        'Portfolio': portfolio_cumulative_returns.values,
        'Benchmark': benchmark_cumulative_returns.reindex(portfolio_returns['Date']).values
    }, index=portfolio_returns['Date'])

    # 4. Plotar os retornos cumulativos
    plt.figure(figsize=(12, 6))
    plt.plot(combined_returns.index, combined_returns['Portfolio'], label='Portfólio')
    plt.plot(combined_returns.index, combined_returns['Benchmark'], label='S&P 500')
    plt.title('Retornos Cumulativos: Portfólio vs S&P 500')
    plt.xlabel('Data')
    plt.ylabel('Retorno Cumulativo')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. Análise de Risco
    # Volatilidade anualizada do portfólio
    portfolio_volatility = portfolio_returns['Portfolio Return'].std() * np.sqrt(252)
    print(f"Volatilidade Anualizada do Portfólio: {portfolio_volatility:.2%}")

    # Volatilidade anualizada do benchmark
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    print(f"Volatilidade Anualizada do S&P 500: {benchmark_volatility:.2%}")

    portfolio_max_drawdown = calculate_max_drawdown(portfolio_cumulative_returns)
    print(f"Máximo Drawdown do Portfólio: {portfolio_max_drawdown:.2%}")

    benchmark_max_drawdown = calculate_max_drawdown(benchmark_cumulative_returns)
    print(f"Máximo Drawdown do S&P 500: {benchmark_max_drawdown:.2%}")

    portfolio_var = calculate_var(portfolio_returns['Portfolio Return'])
    print(f"VaR do Portfólio (95% de confiança): {portfolio_var:.2%}")

    benchmark_var = calculate_var(benchmark_returns)
    print(f"VaR do S&P 500 (95% de confiança): {benchmark_var:.2%}")

    portfolio_annual_return = calculate_annualized_return(portfolio_returns['Portfolio Return'])
    portfolio_sharpe_ratio = (portfolio_annual_return - 0.02) / portfolio_volatility
    print(f"Retorno Anualizado do Portfólio: {portfolio_annual_return:.2%}")
    print(f"Sharpe Ratio do Portfólio: {portfolio_sharpe_ratio:.2f}")

    benchmark_annual_return = calculate_annualized_return(benchmark_returns)
    benchmark_sharpe_ratio = (benchmark_annual_return - 0.02) / benchmark_volatility
    print(f"Retorno Anualizado do S&P 500: {benchmark_annual_return:.2%}")
    print(f"Sharpe Ratio do S&P 500: {benchmark_sharpe_ratio:.2f}")

    portfolio_sortino_ratio = calculate_sortino_ratio(portfolio_returns['Portfolio Return'])
    print(f"Sortino Ratio do Portfólio: {portfolio_sortino_ratio:.2f}")

    benchmark_sortino_ratio = calculate_sortino_ratio(benchmark_returns)
    print(f"Sortino Ratio do S&P 500: {benchmark_sortino_ratio:.2f}")