import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from outils import *
from backtest import *
from metrics import PerformanceMetrics
from backtest import backtest_pso
from optimization import fitness_function
import optuna
import os
import argparse

# TODO: adicionar stress testing em um períodod conturbado
# TODO: usar parâmetros dinâmicos, com base no momento do mercado
# TODO: adicionar o beta do portfólio na fitness
# TODO: filtrar ativos dependendo do momento do mercado
# TODO: precisa pegar os valores certos de imposto e taxa de corretagem, além do risk free rate
# TODO: adicionar a seleção de ações (agora as ações são sempre as mesmas, precisa escolher as ações a cada janela de tempo para poder fazer o backtesting e stress testing)
# TODO: acho que a melhor estratégia é, em cada janela de tempo, identificar diferentes classes de ativos que iremos usar, em momentos de crise, priorizamos aqueles com baixa volatilidade (tipo alguma renda fixa) e nos momentos de alta valorizamos algo mais arriscado (como alguma ação)
# TODO: adicionar algo de retorno à média no cálculo do fitness pode ser interessante (se uma ação está acima da média, penalizar ela)
# ! não sei se o cálculo do retorno tá certo. o pso está acima do sp500 em todos os anos, mas o retorno final é menor
# ! acho que precisa 
# ? melhor usar algo genético?


def main():
    parser = argparse.ArgumentParser(description='Backtest com PSO e Otimização de Hiperparâmetros.')
    parser.add_argument('--force_optimize', action='store_true', help='Força a reotimização dos parâmetros do PSO, mesmo se um arquivo de parâmetros existente for encontrado.')
    args = parser.parse_args()

    # Define os parâmetros
    tickers = [
        # Ações Individuais
        'NVDA', 'BRK-B', 'C', 'JPM', 'AAPL', 
        'GOOG', 'MSFT', 'AMZN', 'TSLA', 'JNJ',
        
        # Índices
        '^GSPC',
        
        # ETFs de Renda Fixa
        'TLT',  # Long-term Treasury ETF
        'IEF',  # 7-10 Year Treasury Bond ETF
        'SHY',  # 1-3 Year Treasury Bond ETF
        'LQD',  # Investment Grade Corporate Bond ETF
        'HYG',  # High Yield Corporate Bond ETF
        
        # ETFs de Commodities
        'GLD',  # Gold ETF
        'SLV',  # Silver ETF
        'DBC',  # Commodity Index ETF
        
        # ETFs de Setores Defensivos
        'XLP',  # Consumer Staples Sector ETF
        'XLU',  # Utilities Sector ETF
        'XLV',  # Health Care Sector ETF
        
        # ETFs Protegidos contra a Inflação
        'TIP',  # TIPS Bond ETF
        
        # ETFs de Baixa Volatilidade e Estratégias Defensivas
        'SPLV',  # Low Volatility ETF
        'USMV',  # Minimum Volatility ETF
        'VIG',   # Dividend Appreciation ETF
    ]

    start_date = '2020-01-01'
    end_date = '2024-04-30'
    initial_investment = 1_000_000

    # Baixa os preços ajustados históricos
    print("Baixando dados históricos...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Remove informações de fuso horário
    data.index = data.index.tz_localize(None)

    # Baixa os dados do benchmark
    benchmark_ticker = '^GSPC'
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
    benchmark_data.index = benchmark_data.index.tz_localize(None)

    # Definindo parâmetros fixos para a função de fitness
    fitness_params = {
        'w_return': 1.0,
        'w_momentum': 0.5,
        'w_deviation': 0.5,
        'w_sortino': 2.0,
        'w_cvar': -1.0,
        'w_drawdown': -1.0,
        'w_transaction_cost': -0.5,
        'w_concentration': -0.5,
        'w_portfolio_volatility': -0.0,
        'alpha': 0.95,
        'risk_free_rate': 0.02
    }

    # Verifica se o arquivo de parâmetros já existe e se a otimização não está sendo forçada
    if os.path.exists(PARAMS_FILE) and not args.force_optimize:
        # Carrega os parâmetros salvos
        best_pso_params = load_params(PARAMS_FILE)
        print(f"Usando parâmetros carregados: {best_pso_params}")
    else:
        # Função de objetivo para o Optuna
        def objective(trial):
            # Sugerindo valores para os hiperparâmetros do PSO
            num_particles = trial.suggest_int('num_particles', 20, 50)
            max_iter = trial.suggest_int('max_iter', 100, 300)
            w = trial.suggest_float('w', 0.5, 1.0)
            c1 = trial.suggest_float('c1', 1.5, 3.0)
            c2 = trial.suggest_float('c2', 1.0, 2.0)

            # Definindo os parâmetros do PSO
            pso_params = {
                'num_particles': num_particles,
                'max_iter': max_iter,
                'w': w,
                'c1': c1,
                'c2': c2
            }

            # Executa o backtest com os parâmetros atuais do PSO
            portfolio_values, allocations_df = backtest_pso(
                data=data,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                fitness_function=fitness_function,
                rebalance_period='1ME',
                initial_investment=initial_investment,
                pso_params=pso_params,
                **fitness_params  # Passa os parâmetros de fitness
            )

            # Define uma métrica de performance para maximizar
            # Por exemplo, o valor final do portfólio
            final_portfolio_value = portfolio_values.iloc[-1]

            # Alternativamente, pode-se usar o Índice de Sharpe ou outra métrica
            # Retorna o valor final do portfólio
            return final_portfolio_value

        # Cria o estudo Optuna
        print("Iniciando a otimização dos hiperparâmetros com Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, timeout=600)  # Ajuste n_trials e timeout conforme necessário

        # Exibe os melhores parâmetros encontrados
        print("\nMelhores parâmetros encontrados:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
        print(f"Melhor valor do portfólio: {study.best_value}")

        # Salva os melhores parâmetros em um arquivo JSON
        save_params(study.best_params, PARAMS_FILE)

        # Define os melhores parâmetros para uso posterior
        best_pso_params = study.best_params

    # Executa o backtest com os melhores parâmetros
    print("\nExecutando o backtest com os melhores parâmetros...")
    portfolio_values, allocations_df = backtest_pso(
        data=data,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        fitness_function=fitness_function,
        rebalance_period='1ME',
        initial_investment=initial_investment,
        pso_params=best_pso_params,
        **fitness_params
    )

    # Salvar os resultados
    print("\nSalvando os resultados...")
    portfolio_values.to_csv('portfolio_values.csv')
    allocations_df.to_csv('allocations.csv')

    # Plotar o valor do portfólio
    print("Gerando gráfico de comparação com o benchmark...")
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Valor do Portfólio')
    plt.plot(benchmark_data, label='Benchmark (^GSPC)')
    plt.legend()
    plt.title('Valor do Portfólio vs Benchmark')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.savefig('portfolio_vs_benchmark.png')
    plt.show()

    print("\nBacktest concluído com sucesso!")

if __name__ == "__main__":
    main()