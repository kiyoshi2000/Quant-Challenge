# main.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

from backtest import backtest_pso
from metrics import PerformanceMetrics
from trend_identifier import MarketTrendIdentifier

# Configuração básica de logging
logging.basicConfig(
    filename='backtest.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_individual_allocations(allocations_df, output_dir='asset_allocations'):
    """
    Plota e salva gráficos individuais para a evolução da alocação percentual de cada ativo.

    Parâmetros:
    - allocations_df: pandas.DataFrame contendo as alocações diárias dos ativos.
    - output_dir: str, diretório onde os gráficos serão salvos.
    """
    # Criar o diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tickers = allocations_df.columns.tolist()
    
    for ticker in tickers:
        plt.figure(figsize=(10, 6))
        plt.plot(allocations_df.index, allocations_df[ticker] * 100, label=f'{ticker} Allocation', color='blue')
        plt.title(f'Evolução da Alocação de {ticker} ao Longo dos Anos')
        plt.xlabel('Data')
        plt.ylabel('Alocação Percentual (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Salvar o gráfico no diretório especificado
        plt.savefig(os.path.join(output_dir, f'{ticker}_allocation_evolution.png'))
        plt.close()  # Fechar a figura para liberar memória

        logging.info(f'Gráfico para {ticker} salvo em {output_dir}/')

def main():
    # Definição dos parâmetros
    
    tickers = [
        'NVDA', 'BRK-B', 'C', 'JPM', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'JNJ',
        'TLT', 'GLD', 'EFA', 'EEM', 'LQD'
    ]



    start_date = '2015-01-01'
    end_date = '2024-04-30'
    initial_investment = 1_000_000

    logging.info("Iniciando o script principal.")

    # Inicializa o identificador de tendência
    trend_identifier = MarketTrendIdentifier(
        tickers=tickers,
        benchmark_ticker='^GSPC',
        start_date=start_date,
        end_date=end_date,
        trend_window=252,
        beta_window=126,
        ma_short=50,
        ma_long=200
    )

    # Executa o processo de identificação de tendência
    logging.info("Executando a identificação da tendência do mercado.")
    weights, params, trend = trend_identifier.run()
    logging.info(f"Tendência identificada: {trend}")

    # Baixar preços ajustados históricos
    logging.info("Baixando preços ajustados históricos para o backtest.")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    logging.info("Preços ajustados baixados com sucesso.")

    # Remover informações de timezone
    data.index = data.index.tz_localize(None)

    # Baixar dados do benchmark
    benchmark_ticker = '^GSPC'
    logging.info("Baixando dados do benchmark.")
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
    benchmark_data.index = benchmark_data.index.tz_localize(None)
    logging.info("Dados do benchmark baixados com sucesso.")

    # Passar os parâmetros ajustados para o backtest
    logging.info("Iniciando o backtest com otimização PSO.")
    portfolio_values, allocations_df = backtest_pso(
        data=data,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        rebalance_period='1ME',  # Rebalanceamento mensal
        initial_investment=initial_investment,
        **params  # Passa os parâmetros ajustados dinamicamente
    )

    # Garantir que o índice de allocations_df é datetime e está ordenado
    allocations_df.index = pd.to_datetime(allocations_df.index)
    allocations_df = allocations_df.sort_index()

    # Estender as alocações para frequência diária preenchendo para frente
    allocations_daily = allocations_df.reindex(data.index, method='ffill')

    # Remover quaisquer valores NaN
    allocations_daily = allocations_daily.fillna(method='ffill').fillna(method='bfill')
    allocations_daily = allocations_daily.div(allocations_daily.sum(axis=1), axis=0)  # Normalizar os pesos

    portfolio_values = portfolio_values.dropna()

    # Calcular retornos cumulativos do benchmark
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() * initial_investment

    # Alinhar os dados do benchmark com as datas do portfólio
    benchmark_cumulative_returns = benchmark_cumulative_returns[benchmark_cumulative_returns.index.isin(portfolio_values.index)]

    # Plotar retornos cumulativos
    logging.info("Plotando retornos cumulativos do portfólio versus o benchmark.")
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values.values, label='Portfólio Otimizado')
    plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns.values, label='Benchmark S&P 500')
    plt.title('Retornos Cumulativos: Portfólio vs. S&P 500')
    plt.xlabel('Data')
    plt.ylabel('Valor do Portfólio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotar as alocações
    logging.info("Plotando alocações de ativos ao longo do tempo.")
    plt.figure(figsize=(12, 6))
    allocations_daily.plot.area(stacked=True, ax=plt.gca())
    plt.title('Alocações do Portfólio ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Alocação Percentual')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()

    # Plotar retornos cumulativos das ações rebalanceadas
    logging.info("Plotando retornos cumulativos das alocações individuais dos ativos.")
    plot_individual_allocations(allocations_daily)
    logging.info("Gráficos individuais de alocação de ativos gerados com sucesso.")

    # Calcular retornos diários
    portfolio_daily_returns = portfolio_values.pct_change().dropna()
    benchmark_daily_returns = benchmark_returns[benchmark_returns.index.isin(portfolio_daily_returns.index)]

    # Inicializar métricas de performance
    metrics = PerformanceMetrics()

    # Calcular métricas de risco para o portfólio
    portfolio_annual_return = metrics.calculate_annualized_return(portfolio_daily_returns)
    portfolio_annual_volatility = metrics.calculate_annualized_volatility(portfolio_daily_returns)
    portfolio_sharpe_ratio = metrics.calculate_sharpe_ratio(portfolio_daily_returns)
    portfolio_max_drawdown = metrics.calculate_max_drawdown(portfolio_values)

    # Calcular métricas de risco para o benchmark
    benchmark_annual_return = metrics.calculate_annualized_return(benchmark_daily_returns)
    benchmark_annual_volatility = metrics.calculate_annualized_volatility(benchmark_daily_returns)
    benchmark_sharpe_ratio = metrics.calculate_sharpe_ratio(benchmark_daily_returns)
    benchmark_max_drawdown = metrics.calculate_max_drawdown(benchmark_cumulative_returns)

    # Imprimir métricas de risco
    logging.info("Calculando e exibindo métricas de risco e desempenho.")
    print("Métricas de Risco e Desempenho:")
    print("\nPortfólio Otimizado:")
    print(f"Retorno Anualizado: {portfolio_annual_return:.2%}")
    print(f"Volatilidade Anualizada: {portfolio_annual_volatility:.2%}")
    print(f"Sharpe Ratio: {portfolio_sharpe_ratio:.2f}")
    print(f"Máximo Drawdown: {portfolio_max_drawdown:.2%}")

    print("\nBenchmark S&P 500:")
    print(f"Retorno Anualizado: {benchmark_annual_return:.2%}")
    print(f"Volatilidade Anualizada: {benchmark_annual_volatility:.2%}")
    print(f"Sharpe Ratio: {benchmark_sharpe_ratio:.2f}")
    print(f"Máximo Drawdown: {benchmark_max_drawdown:.2%}")

    logging.info("Script principal concluído com sucesso.")

if __name__ == "__main__":
    main()
