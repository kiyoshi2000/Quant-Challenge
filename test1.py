import yfinance as yf
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


class PortfolioGenerator:
    def __init__(self, tickers, investment_amount):
        self.tickers = tickers
        self.investment_amount = investment_amount
        self.current_allocation = np.full(len(tickers), investment_amount / len(tickers))

    def download_data(self, start_date, end_date):
        self.data = yf.download(
            self.tickers, start=start_date, end=end_date
        )['Adj Close']
        return self.data

    def generate_chromosome(self):
        """Generates a random chromosome representing asset weights."""
        weights = np.random.random(len(self.tickers))
        return weights / np.sum(weights)  # Normalize to sum to 1

    def create_population(self, population_size):
        """Creates a population of chromosomes."""
        return [self.generate_chromosome() for _ in range(population_size)]


class FitnessEvaluator:
    def __init__(self, data, risk_free_rate=0.02):
        self.data = data
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe_ratio(self, chromosome):
        """Calculates the Sharpe Ratio for a given portfolio."""
        portfolio_returns = np.dot(
            self.data.pct_change().dropna(), chromosome
        )
        std_dev = np.std(portfolio_returns)
        if std_dev == 0:  # Avoid division by zero
            return 0
        sharpe_ratio = (
            np.mean(portfolio_returns) - self.risk_free_rate
        ) / std_dev
        return sharpe_ratio


class GeneticAlgorithm:
    def __init__(
        self,
        population,
        data,
        fitness_evaluator,
        mutation_rate=0.01,
    ):
        self.population = population
        self.data = data
        self.fitness_evaluator = fitness_evaluator
        self.mutation_rate = mutation_rate

    def calculate_fitness(self, chromosome):
        return self.fitness_evaluator.calculate_sharpe_ratio(chromosome)

    def roulette_wheel_selection(self, fitness_values):
        """Selects an individual from the population using the roulette wheel method based on fitness values."""
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1e-6 for f in fitness_values]

        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            raise ValueError("The sum of fitness values is zero, check the fitness function.")

        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_values):
            current += fitness
            if current > pick:
                return self.population[i]

    def single_point_crossover(self, parent1, parent2):
        """Performs a single-point crossover between two parents."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate(
            (parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate(
            (parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutation(self, chromosome):
        """Mutates a chromosome with a given mutation rate and normalizes the result."""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.uniform(0, 1)
        return chromosome / np.sum(chromosome)

    def evolve(self):
        """Evolves the population by performing selection, crossover, and mutation."""
        fitness_values = [
            self.calculate_fitness(chromosome)
            for chromosome in self.population
        ]
        new_population = []
        for _ in range(len(self.population) // 2):
            parent1 = self.roulette_wheel_selection(fitness_values)
            parent2 = self.roulette_wheel_selection(fitness_values)

            if parent1 is None or parent2 is None:
                raise ValueError("Failed to select parents, check the selection function.")

            child1, child2 = self.single_point_crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.extend([child1, child2])

        self.population = new_population


class Rebalancer:
    def __init__(self, portfolio_generator, genetic_algorithm, data, tickers, periods=8):
        self.pg = portfolio_generator
        self.ga = genetic_algorithm
        self.data = data
        self.tickers = tickers
        self.periods = periods
        self.exposure_history = pd.DataFrame(columns=tickers)
        self.allocations = []  # Stores allocations and dates

    def rebalance(self):
        """Rebalances the portfolio over multiple periods based on genetic algorithm optimization."""
        dates = self.data.index
        total_days = len(dates)

        for period in range(self.periods):
            print(f"Rebalancing: Semester {period+1}")
            # Evolves the population and finds the best chromosome (new allocation)
            self.ga.evolve()
            best_chromosome = self.ga.population[0]

            # Updates allocations based on the best chromosome
            total_value = np.sum(self.pg.current_allocation)
            new_allocation = best_chromosome * total_value

            # Creates a new DataFrame for the new row
            new_row = pd.DataFrame([new_allocation], columns=self.tickers, index=[f"Semester {period+1}"])

            # Checks if the new row contains valid values before concatenating
            if not new_row.isna().all().all() and new_row.sum().sum() > 0:
                self.exposure_history = pd.concat([self.exposure_history, new_row])

            # Determines the start and end dates for the period
            period_start_index = period * (total_days // self.periods)
            if period == self.periods - 1:
                period_end_index = total_days - 1
            else:
                period_end_index = (period + 1) * (total_days // self.periods) - 1

            period_start = dates[period_start_index]
            period_end = dates[period_end_index]

            # Stores the allocation and corresponding dates
            self.allocations.append({
                'start': period_start,
                'end': period_end,
                'allocation': new_allocation
            })

            # Updates the current portfolio
            self.pg.current_allocation = new_allocation

    def plot_exposure(self):
        """Plots the portfolio exposure over time using a stacked bar chart with percentages."""
        exposure_percentage = self.exposure_history.div(self.exposure_history.sum(axis=1), axis=0) * 100
        exposure_percentage.plot(kind='bar', stacked=True, figsize=(12, 6))

        plt.xlabel('Semester')
        plt.ylabel('Portfolio Exposure (%)')
        plt.title('Portfolio Exposure Over Time (%)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    def compare_with_gs(self, gs_column_name):
        """Compares the optimized portfolio returns with the GS benchmark returns."""
        portfolio_returns_list = []

        for period_info in self.allocations:
            period_start = period_info['start']
            period_end = period_info['end']
            allocation = period_info['allocation'] / np.sum(period_info['allocation'])  # Normalize allocation

            # Select data for the period
            period_data = self.data.loc[period_start:period_end, self.tickers]
            period_returns = period_data.pct_change().dropna()

            # Calculate portfolio returns for the period
            weighted_returns = (period_returns * allocation).sum(axis=1)
            portfolio_returns_list.append(weighted_returns)

        # Concatenate returns from all periods
        portfolio_returns = pd.concat(portfolio_returns_list)

        # Calculate cumulative returns of the optimized portfolio
        optimized_portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1

        # Calculate cumulative returns of the benchmark (GS)
        gs_returns = self.data[gs_column_name].pct_change().dropna()
        gs_cum_returns = (1 + gs_returns).cumprod() - 1

        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.plot(optimized_portfolio_cum_returns.index, optimized_portfolio_cum_returns.values, label='Optimized Portfolio')
        plt.plot(gs_cum_returns.index, gs_cum_returns.values, label='GS Benchmark', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title('Optimized Portfolio vs GS Benchmark')
        plt.legend()
        plt.show()


# Code usage:
if __name__ == "__main__":
    tickers = ['NVDA', 'BRK-B', 'C', 'JPM', '^VIX']  # Remove GS from balanced assets
    benchmark_ticker = 'GS'  # Just a symbol, not a list
    investment_amount = 100000  # Initial allocation
    start_date = '2020-01-01'
    end_date = '2024-04-30'

    portfolio_generator = PortfolioGenerator(tickers, investment_amount)
    data = portfolio_generator.download_data(start_date, end_date)

    # Download GS data for comparison
    gs_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']

    # Ensure both DataFrames have timezone-naive indexes
    data.index = data.index.tz_localize(None)  # Remove timezone from date index
    gs_data.index = gs_data.index.tz_localize(None)  # Remove timezone from date index

    # Convert gs_data to a DataFrame and rename the column to 'GS'
    gs_data = gs_data.to_frame(name='GS')

    # Safely concatenate the data
    data = pd.concat([data, gs_data], axis=1)  # Add GS to the data for comparison

    # Check the column name of GS
    print("Columns after adding GS:", data.columns)

    population = portfolio_generator.create_population(population_size=50)
    fitness_evaluator = FitnessEvaluator(data[tickers])
    ga = GeneticAlgorithm(population, data[tickers], fitness_evaluator)

    rebalancer = Rebalancer(portfolio_generator, ga, data, tickers, periods=8)
    rebalancer.rebalance()
    rebalancer.plot_exposure()  # Stacked bar chart with percentage exposure

    # Adjust comparison using the correct GS column name
    # rebalancer.compare_with_gs('GS')  # Now GS is a column named 'GS'
