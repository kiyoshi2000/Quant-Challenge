import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

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