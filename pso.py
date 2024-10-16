import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# Função de fitness baseada no Sharpe Ratio
class FitnessEvaluator:
    def __init__(self, data, risk_free_rate=0.02):
        self.data = data
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe_ratio(self, chromosome):
        """Calcula o Sharpe Ratio para um portfólio."""
        portfolio_returns = np.dot(self.data.pct_change().dropna(), chromosome)
        std_dev = np.std(portfolio_returns)
        if std_dev == 0:
            return 0
        sharpe_ratio = (np.mean(portfolio_returns) - self.risk_free_rate) / std_dev
        return sharpe_ratio


# Algoritmo Genético (GA)
class GeneticAlgorithm:
    def __init__(self, population, fitness_evaluator, mutation_rate=0.01):
        self.population = population
        self.fitness_evaluator = fitness_evaluator
        self.mutation_rate = mutation_rate

    def calculate_fitness(self, chromosome):
        return self.fitness_evaluator.calculate_sharpe_ratio(chromosome)

    def roulette_wheel_selection(self, fitness_values):
        """Seleciona um indivíduo da população baseado no fitness."""
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1e-6 for f in fitness_values]

        total_fitness = sum(fitness_values)
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_values):
            current += fitness
            if current > pick:
                return self.population[i]

    def single_point_crossover(self, parent1, parent2):
        """Realiza um crossover entre dois pais."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutation(self, chromosome):
        """Realiza mutação em um cromossomo."""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.uniform(0, 1)
        return chromosome / np.sum(chromosome)

    def evolve(self):
        """Evolui a população por seleção, crossover e mutação."""
        fitness_values = [self.calculate_fitness(chromosome) for chromosome in self.population]
        new_population = []
        for _ in range(len(self.population) // 2):
            parent1 = self.roulette_wheel_selection(fitness_values)
            parent2 = self.roulette_wheel_selection(fitness_values)
            child1, child2 = self.single_point_crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.extend([child1, child2])
        self.population = new_population


# Algoritmo PSO
class PSO:
    def __init__(self, num_particles, num_assets, fitness_evaluator, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.fitness_evaluator = fitness_evaluator
        self.num_particles = num_particles
        self.particles = [self.Particle(num_assets) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].position
        self.global_best_fitness = self.fitness_evaluator.calculate_sharpe_ratio(self.global_best_position)
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    class Particle:
        def __init__(self, num_assets):
            self.position = np.random.random(num_assets)
            self.position /= np.sum(self.position)
            self.velocity = np.zeros(num_assets)
            self.best_position = np.copy(self.position)
            self.best_fitness = -np.inf

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                fitness = self.fitness_evaluator.calculate_sharpe_ratio(particle.position)
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(particle.best_position)

                inertia = self.w * particle.velocity
                cognitive = self.c1 * random.random() * (particle.best_position - particle.position)
                social = self.c2 * random.random() * (self.global_best_position - particle.position)
                particle.velocity = inertia + cognitive + social
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, 0, 1)
                particle.position /= np.sum(particle.position)
        return self.global_best_position


# Gerar a população para GA
class PortfolioGenerator:
    def __init__(self, tickers, population_size):
        self.tickers = tickers
        self.population_size = population_size

    def generate_chromosome(self):
        weights = np.random.random(len(self.tickers))
        return weights / np.sum(weights)

    def create_population(self):
        return [self.generate_chromosome() for _ in range(self.population_size)]

# Função para calcular o retorno esperado do portfólio
def calculate_portfolio_return(weights, returns):
    return np.sum(weights * returns.mean())

# Função para calcular o risco do portfólio
def calculate_portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Função para calcular o Sharpe Ratio
def calculate_sharpe_ratio(portfolio_return, portfolio_risk, risk_free_rate=0.02):
    return (portfolio_return - risk_free_rate) / portfolio_risk

if __name__ == "__main__":
    # Definir os parâmetros
    tickers = ['NVDA', 'BRK-B', 'C', 'JPM']
    start_date = '2020-01-01'
    end_date = '2024-04-30'
    population_size = 50
    investment_amount = 100000

    # Baixar os dados
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Inicializar o avaliador de fitness com os dados
    fitness_evaluator = FitnessEvaluator(data)

    # Gerar a população para o algoritmo genético
    portfolio_generator = PortfolioGenerator(tickers, population_size)
    population = portfolio_generator.create_population()

    # Algoritmo Genético (GA)
    ga = GeneticAlgorithm(population, fitness_evaluator)
    for _ in range(100):  # Evoluir por 100 gerações
        ga.evolve()

    best_ga_allocation = ga.population[0]
    print("Melhor portfólio do GA:", best_ga_allocation)

    # Algoritmo PSO
    pso = PSO(num_particles=30, num_assets=len(tickers), fitness_evaluator=fitness_evaluator)
    best_pso_allocation = pso.optimize()
    print("Melhor portfólio do PSO:", best_pso_allocation)

    # Exibir as alocações
    allocation_df = pd.DataFrame({
        'Tickers': tickers,
        'GA Allocation': best_ga_allocation,
        'PSO Allocation': best_pso_allocation
    })
    print("\nComparação de Alocações:")
    print(allocation_df)

    # Usar dados simulados para calcular retornos e covariância
    returns = data.pct_change().dropna()  # Calcula os retornos diários
    cov_matrix = returns.cov()  # Matriz de covariância dos retornos

    # Calcular retorno esperado e risco para o portfólio do GA
    ga_portfolio_return = calculate_portfolio_return(best_ga_allocation, returns)
    ga_portfolio_risk = calculate_portfolio_risk(best_ga_allocation, cov_matrix)
    ga_sharpe_ratio = calculate_sharpe_ratio(ga_portfolio_return, ga_portfolio_risk)

    # Calcular retorno esperado e risco para o portfólio do PSO
    pso_portfolio_return = calculate_portfolio_return(best_pso_allocation, returns)
    pso_portfolio_risk = calculate_portfolio_risk(best_pso_allocation, cov_matrix)
    pso_sharpe_ratio = calculate_sharpe_ratio(pso_portfolio_return, pso_portfolio_risk)

    print(ga_portfolio_return, ga_portfolio_risk, ga_sharpe_ratio, pso_portfolio_return, pso_portfolio_risk, pso_sharpe_ratio)