import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import norm

MAX_ACCEPTABLE_RISK = 0.15

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

        # Penalize large changes in allocations (optional)
        penalty_allocation_change = 0
        if prev_weights is not None:
            penalty_allocation_change = np.sum(np.abs(weights - prev_weights)) * 0.01

        # Penalize if portfolio risk exceeds maximum acceptable risk
        global MAX_ACCEPTABLE_RISK
        penalty_risk = 0
        if portfolio_risk > MAX_ACCEPTABLE_RISK:
            penalty_risk = (portfolio_risk - MAX_ACCEPTABLE_RISK) * 100  # Adjust multiplier as needed

        # Total penalty
        total_penalty = penalty_allocation_change + penalty_risk

        # Adjusted fitness
        adjusted_fitness = sharpe_ratio - total_penalty

        # Penalizar grandes mudanças nas alocações
        # if prev_weights is not None:
        #     penalty = np.sum(np.abs(weights - prev_weights))  # Soma das diferenças absolutas entre alocações
        #     sharpe_ratio -= penalty * 0.01  # Ajuste o multiplicador conforme necessário

        return adjusted_fitness

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

                # Normalize first
                particle.position /= np.sum(particle.position)
                # Then clip
                particle.position = np.clip(particle.position, 0, 1)
                # Re-normalize if necessary
                particle.position /= np.sum(particle.position)

        return self.global_best_position
