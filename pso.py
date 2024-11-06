import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import norm

MAX_ACCEPTABLE_RISK = 0.15

# Particle Swarm Optimization (PSO) Algorithm
class PSO:
    def __init__(self, num_particles, num_assets, max_iter=200, w=0.7, c1=1.7, c2=2):
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

    def fitness_function(self, weights, returns, cov_matrix, prev_weights, risk_free_rate=0.02):
        """
        Calculate the fitness of a particle (portfolio allocation).
        Includes penalties for large allocation changes and exceeding risk limits.
        """
        # Annualized expected return
        annualized_return = np.sum(weights * returns.mean()) * 252

        # Annualized covariance matrix
        annualized_cov_matrix = cov_matrix * 252

        # Portfolio risk (standard deviation)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(annualized_cov_matrix, weights)))

        # Calculate Sharpe Ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_risk

        # Penalize large changes in allocations (optional)
        penalty_allocation_change = 0
        if prev_weights is not None:
            # Sum of absolute differences between current and previous weights
            penalty_allocation_change = np.sum(np.abs(weights - prev_weights)) * 0.01  # Adjust multiplier as needed

        # Penalize if portfolio risk exceeds maximum acceptable risk
        penalty_risk = 0
        MAX_ACCEPTABLE_RISK = 0.15  # Example: 15% annualized volatility
        if portfolio_risk > MAX_ACCEPTABLE_RISK:
            # Penalty proportional to the excess risk
            penalty_risk = (portfolio_risk - MAX_ACCEPTABLE_RISK) * 100  # Adjust multiplier as needed

        # Total penalty
        total_penalty = penalty_allocation_change + penalty_risk

        # Adjusted fitness value
        adjusted_fitness = sharpe_ratio - total_penalty

        return adjusted_fitness

    def optimize(self, returns, cov_matrix, prev_weights=None):
        """
        Optimize the portfolio weights using PSO.
        """
        for _ in range(self.max_iter):
            for particle in self.particles:
                # Calculate fitness of the particle
                fitness = self.fitness_function(
                    particle.position, returns, cov_matrix, prev_weights
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
