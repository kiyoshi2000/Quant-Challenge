import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
# Value at Risk (VaR)
from scipy.stats import norm

# Algoritmo PSO
class PSO:
    def __init__(self, num_particles, num_assets, max_iter=200, w=0.7, c1=1.7, c2=2):
        self.num_particles = num_particles
        self.particles = [self.Particle(num_assets) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].position
        self.global_best_fitness = -np.inf  # Inicializa com o pior valor possível
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    class Particle:
        def __init__(self, num_assets):
            self.position = np.random.random(num_assets)
            self.position /= np.sum(self.position)  # Normaliza as posições para somarem 1
            self.velocity = np.zeros(num_assets)
            self.best_position = np.copy(self.position)
            self.best_fitness = -np.inf

    def sharpe_ratio(self, weights, returns, cov_matrix, prev_weights, risk_free_rate=0.02):
        annualized_return = np.sum(weights * returns.mean()) * 252
        annualized_cov_matrix = cov_matrix * 252
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(annualized_cov_matrix, weights)))
        sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_risk

        # Penalizar grandes mudanças nas alocações
        if prev_weights is not None:
            penalty = np.sum(np.abs(weights - prev_weights))  # Soma das diferenças absolutas entre alocações
            sharpe_ratio -= penalty * 0.01  # Ajuste o multiplicador conforme necessário

        return sharpe_ratio

    def optimize(self, returns, cov_matrix, prev_weights=None):
        for _ in range(self.max_iter):
            for particle in self.particles:
                # Calcula o Sharpe Ratio com penalização para grandes variações
                fitness = self.sharpe_ratio(particle.position, returns, cov_matrix, prev_weights)

                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(particle.best_position)

                # Atualiza a velocidade e a posição das partículas
                inertia = self.w * particle.velocity
                cognitive = self.c1 * random.random() * (particle.best_position - particle.position)
                social = self.c2 * random.random() * (self.global_best_position - particle.position)
                particle.velocity = inertia + cognitive + social
                particle.position += particle.velocity

                # Impor restrições de variação
                if prev_weights is not None:
                    particle.position = np.clip(particle.position, prev_weights - 0.1, prev_weights + 0.1)  # Limitar a mudança para 10%

                particle.position = np.clip(particle.position, 0, 1)  # Allow weights between 0% and 100%  # Limitar cada ativo entre 5% e 50%
                # Normalize first
                particle.position /= np.sum(particle.position)
                # Then clip
                particle.position = np.clip(particle.position, 0, 1)
                # Re-normalize if necessary
                particle.position /= np.sum(particle.position)

        return self.global_best_position

# Função para calcular o retorno esperado do portfólio
def calculate_portfolio_return(weights, returns):
    return np.sum(weights * returns.mean())

# Função para calcular o risco do portfólio
def calculate_portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Função para calcular o Sharpe Ratio
def calculate_sharpe_ratio(portfolio_return, portfolio_risk, risk_free_rate=0.02):
    return (portfolio_return - risk_free_rate) / portfolio_risk

# Função de backtest com o PSO
def backtest_pso(data, tickers, start_date, end_date, rebalance_period='3ME'):
    results = []
    portfolio_allocations = []
    prev_weights = None  # Inicialmente, não há pesos anteriores

    # Converte o índice de data para formato de datetime
    data = data.loc[start_date:end_date]
    data.index = data.index.tz_localize(None)

    # Inicializando a data de rebalanceamento
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_period)

    for date in rebalance_dates:
        # Usar apenas dados anteriores à data de rebalanceamento
        historical_data = data[data.index <= date]

        # Calcular retornos e matriz de covariância
        returns = historical_data.pct_change().dropna()
        cov_matrix = returns.cov()

        # Algoritmo PSO, usando prev_weights para suavizar as alocações
        pso = PSO(num_particles=30, num_assets=len(tickers))
        best_pso_allocation = pso.optimize(returns=returns, cov_matrix=cov_matrix, prev_weights=prev_weights)

        # Armazena a alocação de portfólio e a data
        portfolio_allocations.append((date, best_pso_allocation))

        # Calcular o retorno do portfólio no período subsequente
        future_data = data[(data.index > date) & (data.index <= date + timedelta(days=90))]
        future_returns = future_data.pct_change().dropna().mean()
        portfolio_return = np.dot(best_pso_allocation, future_returns)

        # Armazena o retorno e a data
        results.append((date, portfolio_return))

        # Atualiza prev_weights para o próximo período
        prev_weights = best_pso_allocation

    return pd.DataFrame(results, columns=['Date', 'Portfolio Return']), pd.DataFrame(portfolio_allocations, columns=['Date', 'Allocation'])
def plot_portfolio_allocations(portfolio_allocations, tickers):
    # Converter as alocações em um DataFrame separado por colunas (uma para cada ativo)
    allocations_df = pd.DataFrame(
        portfolio_allocations['Allocation'].to_list(), 
        columns=tickers, 
        index=portfolio_allocations['Date']
    )

    # Plotar as alocações ao longo do tempo
    plt.figure(figsize=(10, 6))
    
    for ticker in tickers:
        plt.plot(allocations_df.index, allocations_df[ticker], label=ticker)

    plt.title('Pesos do Portfólio ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Alocação (%)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Máximo Drawdown
def calculate_max_drawdown(cumulative_returns):
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_var(returns, confidence_level=0.95):
    mean = returns.mean()
    std_dev = returns.std()
    var = norm.ppf(1 - confidence_level, mean, std_dev)
    return var

# Retornos Anualizados e Sharpe Ratios
def calculate_annualized_return(daily_returns):
    return (1 + daily_returns.mean()) ** 252 - 1

# Sortino Ratios
def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    # Retorno anualizado
    annual_return = calculate_annualized_return(returns)
    # Desvio padrão dos retornos negativos
    downside_returns = returns[returns < 0]
    expected_downside = (downside_returns ** 2).mean()
    downside_deviation = np.sqrt(expected_downside) * np.sqrt(252)
    # Sortino Ratio
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
    return sortino_ratio

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
    portfolio_returns, portfolio_allocations = backtest_pso(data, tickers, start_date, end_date, rebalance_period='3M')

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