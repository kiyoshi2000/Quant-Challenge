import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import norm
from transaction import TransactionCostCalculator, TaxCalculator
from outils import *

# Particle Swarm Optimization (PSO) Algorithm
class PSO:
    def __init__(self, num_assets, num_particles=30, max_iter=67, w=0.5, c1=2.56, c2=2.38):
        self.num_particles = num_particles  # Number of particles in the swarm
        self.particles = [self.Particle(num_assets) for _ in range(num_particles)]  # List of particles
        self.global_best_position = self.particles[0].position  # Global best position (best solution found)
        self.global_best_fitness = -np.inf  # Global best fitness (objective function value)
        self.max_iter = max_iter  # Maximum number of iterations
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive coefficient (particle's own experience)
        self.c2 = c2    # Social coefficient (swarm's experience)

    class Particle:
        def __init__(self, num_assets):
            self.position = np.random.random(num_assets)  # Particle's position (portfolio weights)
            self.position /= np.sum(self.position)        # Normalize weights to sum to 1
            self.velocity = np.zeros(num_assets)          # Particle's velocity
            self.best_position = np.copy(self.position)   # Particle's best known position
            self.best_fitness = -np.inf                   # Particle's best known fitness

    def optimize(self, returns, cov_matrix, portfolio_value, portfolio_values, prev_weights, **kwargs):
        """
        Optimize the portfolio weights using PSO.
        """
        for _ in range(self.max_iter):
            for particle in self.particles:
                # Ensure particle.position (weights) sum to 1 and are within bounds
                particle.position = np.clip(particle.position, 0, 1)
                particle.position /= np.sum(particle.position)
                
                # Calculate fitness of the particle
                fitness = fitness_function(
                    weights=particle.position,
                    returns=returns,
                    cov_matrix=cov_matrix,
                    prev_weights=prev_weights,
                    portfolio_value=portfolio_value,
                    portfolio_values=portfolio_values,
                    **kwargs
                )

                # Update personal best if current fitness is better
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)

                # Update global best if current fitness is better
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(particle.best_position)

                # Update velocity and position of the particle
                inertia = self.w * particle.velocity
                cognitive = self.c1 * random.random() * (particle.best_position - particle.position)
                social = self.c2 * random.random() * (self.global_best_position - particle.position)
                particle.velocity = inertia + cognitive + social
                particle.position += particle.velocity

                # Impose allocation constraints
                if prev_weights is not None:
                    # Limit changes to within ±10% of previous weights
                    particle.position = np.clip(particle.position, prev_weights - 0.3, prev_weights + 0.3)

                # Ensure weights are between 0 and 1
                particle.position = np.clip(particle.position, 0, 0.5)
                # Normalize weights to sum to 1
                particle.position /= np.sum(particle.position)

        return self.global_best_position
    
def fitness_function(weights, returns, cov_matrix, prev_weights, portfolio_value, portfolio_values, 
                    transaction_calculator=TransactionCostCalculator, **kwargs):
    """
    Calculate the fitness of a portfolio based on the specified method with normalized metrics.

    Parameters:
    - weights: np.array of portfolio weights.
    - returns: pd.DataFrame of historical returns.
    - cov_matrix: pd.DataFrame of the covariance matrix of returns.
    - prev_weights: np.array of previous portfolio weights.
    - portfolio_value: Current portfolio value.
    - portfolio_values: pd.Series of portfolio values over time.
    - transaction_calculator: Instance to calculate transaction costs.
    - method: String specifying the method ('sharpe', 'sortino', 'cvar', 'max_drawdown', 'custom').
    - scaling_factors: Dict with scaling factors for normalization.
    - **kwargs: Additional arguments required for specific methods.

    Returns:
    - fitness: Calculated fitness value (higher is better).
    """

    scaling_factors = load_scaling_factors()

    # Ensure weights are a numpy array
    weights = np.array(weights)

    transaction_calculator = TransactionCostCalculator()

    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    expected_return = np.mean(portfolio_returns) * 252  # Annualized expected return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # Annualized volatility
    risk_free_rate = kwargs.get('risk_free_rate', 0.02)

    # Custom fitness function combining multiple metrics
    # Weights for each metric (adjust as needed)
    w_return = kwargs.get('w_return', 0.0)
    w_momentum = kwargs.get('w_momentum', 0.0)
    w_deviation = kwargs.get('w_deviation', 0.0)
    w_sortino = kwargs.get('w_sortino', 0.0)

    w_cvar = kwargs.get('w_cvar', -1.0)
    w_drawdown = kwargs.get('w_drawdown', -0.0)
    w_transaction_cost = kwargs.get('w_transaction_cost', 0.0)
    w_concentration = kwargs.get('w_concentration', -0.0)
    w_portfolio_volatility = kwargs.get('w_portfolio_volatility', -1.0)

    # Calculando métricas de momentum e retorno à média
    momentum_window = kwargs.get('momentum_window', 63)  # 3 meses
    momentum = calculate_momentum(returns, window=momentum_window).iloc[-1]

    # Cálculo do desvio do portfólio
    deviation = calculate_portfolio_deviation(portfolio_values)

    # Calculate additional metrics
    # CVaR
    alpha = kwargs.get('alpha', 0.95)
    cvar = calculate_cvar(portfolio_returns, alpha)

    # Calculate concentration
    concentration = np.sqrt(np.sum((weights - 1/len(weights))**2))

    # Calculate cost
    # Calculate the dollar allocation for both previous and new weights
    prev_allocation = prev_weights * portfolio_value
    new_allocation = weights * portfolio_value
    shares_diff = (new_allocation - prev_allocation) / returns.iloc[-1]  # Dividing by the latest prices

    # Calculate transaction cost based on shares_diff
    transaction_cost = transaction_calculator.calculate_total(shares_diff)

    # Maximum Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Downside Deviation for Sortino Ratio
    target_return = kwargs.get('target_return', 0)
    downside_returns = np.minimum(0, portfolio_returns - target_return)
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    sortino_ratio = (expected_return - risk_free_rate) / downside_deviation

    # Normalizar as métricas
    normalized_expected_return = expected_return / scaling_factors.get('expected_return', 1)
    normalized_momentum = momentum.sum() / scaling_factors.get('momentum', 1)
    normalized_deviation = deviation / scaling_factors.get('deviation', 1)
    normalized_sortino = sortino_ratio / scaling_factors.get('sortino_ratio', 1)
    normalized_portfolio_volatility = portfolio_volatility / scaling_factors.get('portfolio_volatility', 1)
    normalized_cvar = cvar / scaling_factors.get('cvar', 1)
    normalized_max_drawdown = max_drawdown / scaling_factors.get('max_drawdown', 1)
    normalized_concentration = concentration / scaling_factors.get('concentration', 1)
    normalized_transaction_cost = transaction_cost / scaling_factors.get('transaction_cost', 1)

    # Fitness function combinando métricas normalizadas
    fitness = (
        w_return * normalized_expected_return +
        w_momentum * normalized_momentum +  # Soma dos momentos dos ativos
        w_deviation * normalized_deviation + 
        w_sortino * normalized_sortino + 
        w_portfolio_volatility * normalized_portfolio_volatility + 
        w_cvar * (-normalized_cvar) +  # Minimizar CVaR
        w_drawdown * (-normalized_max_drawdown) +  # Minimizar drawdown
        w_concentration * normalized_concentration +  # Minimizar concentração
        w_transaction_cost * normalized_transaction_cost  # Minimizar custos de transação
    )

    return fitness

def calculate_cvar(returns, alpha=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR) of a returns series.

    Parameters:
    - returns: pd.Series of portfolio returns.
    - alpha: Confidence level (default is 0.95).

    Returns:
    - cvar: Conditional Value at Risk.
    """
    # Ensure returns are sorted
    sorted_returns = np.sort(returns)
    index = int((1 - alpha) * len(sorted_returns))
    # CVaR is the average of losses beyond VaR
    cvar = np.mean(sorted_returns[:index])
    return cvar

# Taxa de Retorno (Return Rate): Retorno acumulado em um período específico.
def calculate_momentum(returns, window=63):  # 63 dias ≈ 3 meses
    """
    Calcula o retorno acumulado (momentum) sobre uma janela de tempo específica.

    Parameters:
    - returns: pd.DataFrame ou pd.Series de retornos diários.
    - window: Número de períodos (dias) para calcular o momentum.

    Returns:
    - momentum: pd.DataFrame ou pd.Series com o retorno acumulado.
    """
    # Calcula o retorno acumulado multiplicando os retornos diários
    momentum = (1 + returns).rolling(window=window).apply(np.prod, raw=True) - 1
    return momentum

# Índice de Força Relativa (RSI - Relative Strength Index): Mede a velocidade e a mudança dos movimentos de preço.
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Desvio Padrão do Retorno (Volatilidade): Mede a dispersão dos retornos em relação à média.
def calculate_volatility(returns, window=63):
    volatility = returns.rolling(window=window).std()
    return volatility

# Bandas de Bollinger (Bollinger Bands): Usam médias móveis e desvios padrão para identificar níveis de sobrecompra ou sobrevenda.
def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_portfolio_deviation(portfolio_values, window=90):
    """
    Calcula a porcentagem de desvio do portfólio em relação à sua média móvel.
    
    Parâmetros:
    - portfolio_values: pd.Series do valor do portfólio ao longo do tempo.
    - window: Janela de tempo para calcular a média móvel (p.ex., 252 dias para 1 ano).
    
    Retorna:
    - deviation: Valor percentual do desvio em relação à média móvel.
    """
    rolling_mean = portfolio_values.rolling(window=window).mean()
    deviation = (portfolio_values.iloc[-1] - rolling_mean.iloc[-1]) / rolling_mean.iloc[-1]
    return deviation