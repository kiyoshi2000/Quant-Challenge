import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import norm

MAX_ACCEPTABLE_RISK = 0.1
MAX_ACCEPTABLE_DRAWDOWN = -0.2 

# Particle Swarm Optimization (PSO) Algorithm
class PSO:
    def __init__(self, fitness_function, num_assets, num_particles=50, max_iter=10, w=1, c1=1.2, c2=1.2):
        self.num_particles = num_particles  # Number of particles in the swarm
        self.particles = [self.Particle(num_assets) for _ in range(num_particles)]  # List of particles
        self.global_best_position = self.particles[0].position  # Global best position (best solution found)
        self.global_best_fitness = -np.inf  # Global best fitness (objective function value)
        self.max_iter = max_iter  # Maximum number of iterations
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive coefficient (particle's own experience)
        self.c2 = c2    # Social coefficient (swarm's experience)
        self.fitness_function = fitness_function
    class Particle:
        def __init__(self, num_assets):
            self.position = np.random.random(num_assets)  # Particle's position (portfolio weights)
            self.position /= np.sum(self.position)        # Normalize weights to sum to 1
            self.velocity = np.zeros(num_assets)          # Particle's velocity
            self.best_position = np.copy(self.position)   # Particle's best known position
            self.best_fitness = -np.inf                   # Particle's best known fitness

    def optimize(self, returns, cov_matrix, prev_weights=None):
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
                    particle.position, returns, cov_matrix, method=self.fitness_function,
                    w_return=1.0,
                    w_volatility=1.0,
                    w_cvar=2.0,
                    w_drawdown=2.0,
                    w_sortino=-1.0,
                    alpha=0.95,
                    target_return=0,
                    risk_free_rate=0.02
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
                    particle.position = np.clip(particle.position, prev_weights - 0.1, prev_weights + 0.1)

                # Ensure weights are between 0 and 1
                particle.position = np.clip(particle.position, 0, 1)
                # Normalize weights to sum to 1
                particle.position /= np.sum(particle.position)

        return self.global_best_position
    
def fitness_function(weights, returns, cov_matrix, method, **kwargs):
    """
    Calculate the fitness of a portfolio based on the specified method.

    Parameters:
    - weights: np.array of portfolio weights.
    - returns: pd.DataFrame of historical returns.
    - cov_matrix: pd.DataFrame of the covariance matrix of returns.
    - method: String specifying the method ('sharpe', 'sortino', 'cvar', 'max_drawdown', 'custom').
    - **kwargs: Additional arguments required for specific methods.

    Returns:
    - fitness: Calculated fitness value (lower is better).
    """
    # Ensure weights are a numpy array
    weights = np.array(weights)

    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    expected_return = np.mean(portfolio_returns) * 252  # Annualized expected return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # Annualized volatility

    risk_free_rate = kwargs.get('risk_free_rate', 0.02)

    if method == 'sharpe':
        # Calculate Sharpe Ratio
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility
        fitness = -sharpe_ratio  # Negative because we maximize Sharpe Ratio

    elif method == 'sortino':
        # Calculate downside deviation
        target_return = kwargs.get('target_return', 0)
        downside_returns = np.minimum(0, portfolio_returns - target_return)
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)  # Annualized
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation
        fitness = -sortino_ratio  # Negative because we maximize Sortino Ratio

    elif method == 'cvar':
        # Calculate Conditional Value at Risk (CVaR)
        alpha = kwargs.get('alpha', 0.95)
        cvar = calculate_cvar(portfolio_returns, alpha)
        fitness = cvar  # We aim to minimize CVaR

    elif method == 'max_drawdown':
        # Calculate Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        fitness = abs(max_drawdown)  # We aim to minimize max drawdown

    elif method == 'custom':
        # Custom fitness function combining multiple metrics
        # Weights for each metric (adjust as needed)
        w_return = kwargs.get('w_return', 1.0)
        w_volatility = kwargs.get('w_volatility', 1.0)
        w_cvar = kwargs.get('w_cvar', 2.0)
        w_drawdown = kwargs.get('w_drawdown', 2.0)
        w_sortino = kwargs.get('w_sortino', -1.0)  # Negative because we maximize Sortino Ratio

        # Calculate additional metrics
        # CVaR
        alpha = kwargs.get('alpha', 0.95)
        cvar = calculate_cvar(portfolio_returns, alpha)

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

        # Fitness function combining multiple metrics
        fitness = (w_volatility * portfolio_volatility +
                w_cvar * cvar +
                # w_drawdown * abs(max_drawdown) +
                # w_sortino * sortino_ratio -
                w_return * expected_return)

    else:
        raise ValueError("Invalid method specified. Choose 'sharpe', 'sortino', 'cvar', 'max_drawdown', or 'custom'.")

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
    # VaR at the given confidence level
    var = sorted_returns[index]
    # CVaR is the average of losses beyond VaR
    cvar = -np.mean(sorted_returns[:index])
    return cvar
