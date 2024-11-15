import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import json
import os

PARAMS_FILE = 'best_pso_params.json'
SCALING_FACTORS_FILE = 'scaling_factors.json'

# Function to update holdings after transactions
def update_holdings(shares_held, purchase_price, shares_change, current_price):
    """
    Update the number of shares held and the average purchase price after buying or selling shares.

    Parameters:
    - shares_held: Current number of shares held.
    - purchase_price: Current average purchase price per share.
    - shares_change: Number of shares bought (>0) or sold (<0).
    - current_price: Current market price per share.

    Returns:
    - new_shares_held: Updated number of shares held.
    - new_purchase_price: Updated average purchase price per share.
    """
    if shares_change > 0:
        # Buying shares
        total_cost_existing = purchase_price * shares_held
        total_cost_new = current_price * shares_change
        total_shares = shares_held + shares_change
        if total_shares > 0:
            new_purchase_price = (total_cost_existing + total_cost_new) / total_shares
        else:
            new_purchase_price = 0.0
        new_shares_held = total_shares
    elif shares_change < 0:
        # Selling shares
        new_shares_held = max(shares_held + shares_change, 0)  # Ensure shares held don't go negative
        new_purchase_price = purchase_price  # Purchase price remains the same when selling
    else:
        # No change in shares
        new_shares_held = shares_held
        new_purchase_price = purchase_price

    return new_shares_held, new_purchase_price

def update_holdings_and_prices(prev_weights, current_weights, prices, shares_held, purchase_prices):
    """
    Update holdings and purchase prices after rebalancing.
    """
    portfolio_value = sum(shares_held[ticker] * prices[ticker] for ticker in prices.index)
    desired_values = current_weights * portfolio_value
    desired_shares = desired_values / prices
    shares_diff = desired_shares - np.array([shares_held[ticker] for ticker in prices.index])
    shares_diff = pd.Series(shares_diff, index=prices.index)  # Convert to Series for consistent indexing

    for i in range(len(prices.index)):
        ticker = prices.index[i]
        shares_change = shares_diff.iloc[i]
        current_price = prices[ticker]
        purchase_price = purchase_prices[ticker]
        shares_current = shares_held[ticker]

        # Update holdings and purchase price
        new_shares_held, new_purchase_price = update_holdings(
            shares_current, purchase_price, shares_change, current_price
        )
        shares_held[ticker] = new_shares_held
        purchase_prices[ticker] = new_purchase_price

    return shares_held, purchase_prices

# Function to calculate transaction costs
def calculate_transaction_costs(shares_change):
    """
    Calculate the transaction costs for buying or selling shares.

    Parameters:
    - shares_change: Number of shares bought (>0) or sold (<0).

    Returns:
    - commission: Transaction cost in dollars.
    """
    commission_per_share = 0.005  # Commission per share in dollars
    min_commission = 1.0          # Minimum commission per trade
    commission = max(commission_per_share * abs(shares_change), min_commission)
    return commission

# Function to calculate capital gains tax
def calculate_capital_gains_tax(shares_sold, sale_price, purchase_price, tax_rate=0.25):
    """
    Calculate the tax liability from selling shares.

    Parameters:
    - shares_sold: Number of shares sold.
    - sale_price: Sale price per share.
    - purchase_price: Original purchase price per share.
    - tax_rate: Capital gains tax rate (default is 25%).

    Returns:
    - tax_liability: Tax owed on the capital gain.
    """
    sale_proceeds = shares_sold * sale_price
    cost_basis = shares_sold * purchase_price
    capital_gain = sale_proceeds - cost_basis
    # Tax liability is zero if there is a capital loss
    tax_liability = capital_gain * tax_rate if capital_gain > 0 else 0
    return tax_liability

def calculate_transaction_costs_and_taxes(prev_weights, current_weights, prices, portfolio_value, shares_held, purchase_prices):
    """
    Calculate transaction costs and tax liabilities when rebalancing the portfolio.
    """
    if prev_weights is None:
        # No transaction costs or taxes on initial investment
        transaction_costs = 0.0
        tax_liability = 0.0
    else:
        # Calculate changes in holdings
        desired_values = current_weights * portfolio_value
        desired_shares = desired_values / prices
        shares_diff = desired_shares - np.array([shares_held[ticker] for ticker in prices.index])
        shares_diff = pd.Series(shares_diff, index=prices.index)  # Convert to Series for consistent indexing

        # Transaction costs
        commission_per_share = 0.005  # Commission per share in dollars
        min_commission = 1.0          # Minimum commission per trade
        transaction_costs = 0.0
        tax_liability = 0.0

        for i in range(len(prices.index)):
            ticker = prices.index[i]
            shares_change = shares_diff.iloc[i]
            current_price = prices[ticker]
            purchase_price = purchase_prices[ticker]
            shares_current = shares_held[ticker]

            # Calculate transaction cost
            commission = max(commission_per_share * abs(shares_change), min_commission)
            transaction_costs += commission

            # Calculate tax liability if selling shares
            if shares_change < 0:
                shares_sold = -shares_change
                tax_liability += calculate_capital_gains_tax(
                    shares_sold, current_price, purchase_price
                )

    return transaction_costs, tax_liability

# Function to calculate the portfolio return over a future period
def calculate_portfolio_return(allocation, future_returns):
    """
    Calculate the portfolio return over the future period.

    Parameters:
    - allocation: Array of portfolio weights.
    - future_returns: DataFrame of future returns for each asset.

    Returns:
    - period_return: Portfolio return over the period.
    """
    # Average returns over the future period
    mean_future_returns = future_returns.mean()
    # Number of periods (e.g., days)
    num_periods = len(future_returns)
    # Portfolio return calculation
    period_return = np.dot(allocation, mean_future_returns) * num_periods
    return period_return

def calculate_annualized_return(returns):
    """
    Calculate the annualized return.
    """
    cumulative_return = (1 + returns).prod()
    num_periods = len(returns)
    annualized_return = cumulative_return ** (252 / num_periods) - 1
    return annualized_return

def calculate_annualized_volatility(returns):
    """
    Calculate the annualized volatility.
    """
    return returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio.
    """
    excess_returns = returns - (risk_free_rate / 252)
    return (excess_returns.mean() / returns.std()) * np.sqrt(252)

def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown.
    """
    cumulative_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / cumulative_max - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_transaction_costs_and_taxes(prev_weights, current_weights, prices, portfolio_value, shares_held, purchase_prices):
    """
    Calculate transaction costs and tax liabilities when rebalancing the portfolio.
    """
    if prev_weights is None:
        # No transaction costs or taxes on initial investment
        transaction_costs = 0.0
        tax_liability = 0.0
    else:
        # Calculate changes in holdings
        desired_values = current_weights * portfolio_value
        desired_shares = desired_values / prices
        shares_diff = desired_shares - np.array([shares_held[ticker] for ticker in prices.index])

        # Transaction costs
        commission_per_share = 0.005  # Commission per share in dollars
        min_commission = 1.0          # Minimum commission per trade
        transaction_costs = 0.0
        tax_liability = 0.0

        for i, ticker in enumerate(prices.index):
            shares_change = shares_diff.iloc[i]
            current_price = prices[ticker]
            purchase_price = purchase_prices[ticker]
            shares_current = shares_held[ticker]

            # Calculate transaction cost
            commission = max(commission_per_share * abs(shares_change), min_commission)
            transaction_costs += commission

            # Calculate tax liability if selling shares
            if shares_change < 0:
                shares_sold = -shares_change
                tax_liability += calculate_capital_gains_tax(
                    shares_sold, current_price, purchase_price
                )

    return transaction_costs, tax_liability

def save_params(params, filename):
    """
    Salva os parâmetros em um arquivo JSON.
    
    Parameters:
    - params: Dicionário de parâmetros a serem salvos.
    - filename: Nome do arquivo onde os parâmetros serão salvos.
    """
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def load_params(filename):
    """
    Carrega os parâmetros de um arquivo JSON.
    
    Parameters:
    - filename: Nome do arquivo de onde os parâmetros serão carregados.
    
    Returns:
    - params: Dicionário de parâmetros carregados.
    """
    with open(filename, 'r') as f:
        params = json.load(f)
    return params

def load_scaling_factors(filename=SCALING_FACTORS_FILE):
    """
    Carrega os fatores de escala de um arquivo JSON.
    
    Parameters:
    - filename: Nome do arquivo de onde os fatores de escala serão carregados.
    
    Returns:
    - scaling_factors: Dicionário com os fatores de escala.
    """

    # Verifica se os fatores de escala já existem
    if os.path.exists(SCALING_FACTORS_FILE):
        with open(filename, 'r') as f:
            scaling_factors = json.load(f)
    else:
        # Defina os fatores de escala com base na análise histórica
        scaling_factors = {
            "expected_return": 0.3,            # 30% é considerado um retorno alto anualizado
            "momentum": 0.5,                   # 50% é um retorno acumulado alto em 3 meses
            "deviation": 0.1,                  # 10% desvio do portfólio em relação à média
            "sortino_ratio": 3.0,              # Índice de Sortino alto
            "portfolio_volatility": 0.4,       # 40% volatilidade anualizada
            "cvar": 0.2,                        # 20% CVaR
            "max_drawdown": 0.3,                # 30% máximo drawdown
            "concentration": 1.0,               # Medida de concentração normalizada
            "transaction_cost": 10000           # Custos de transação em unidades monetárias
        }
        # Salva os fatores de escala para uso futuro
        save_scaling_factors(scaling_factors, SCALING_FACTORS_FILE)
    
    return scaling_factors

def save_scaling_factors(scaling_factors, filename=SCALING_FACTORS_FILE):
    """
    Salva os fatores de escala em um arquivo JSON.
    
    Parameters:
    - scaling_factors: Dicionário com os fatores de escala a serem salvos.
    - filename: Nome do arquivo onde os fatores serão salvos.
    """
    with open(filename, 'w') as f:
        json.dump(scaling_factors, f, indent=4)

def plot_returns(portfolio_values, benchmark_cumulative_returns):
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values.values, label='Optimized Portfolio')
    plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns.values, label='S&P 500 Benchmark')
    plt.title('Cumulative Returns: Portfolio vs. S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_portfolio_composition(allocations_daily):
    # Plot the allocations
    plt.figure(figsize=(12, 6))
    plt.plot(allocations_daily, label = allocations_daily.columns)
    plt.title('Portfolio Allocations Over Time')
    plt.xlabel('Date')
    plt.ylabel('Allocation Percentage')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()