import pandas as pd
import numpy as np
from portfolio import Portfolio
from transaction import TransactionCostCalculator, TaxCalculator
from optimization import PSO

def backtest_pso(data, tickers, start_date, end_date, rebalance_period='3M', initial_investment=100000):
    portfolio = Portfolio(tickers)
    transaction_calculator = TransactionCostCalculator()
    tax_calculator = TaxCalculator()

    portfolio_value = initial_investment
    portfolio_values = pd.Series(index=data.index, dtype=float)
    portfolio_values.iloc[0] = portfolio_value
    prev_weights = None
    current_weights = np.array([1 / len(tickers)] * len(tickers))  # Start with equal weights

    # Generate rebalance dates and ensure they are trading days
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_period)
    rebalance_dates = [date for date in rebalance_dates if date in data.index]

    for i in range(1, len(data.index)):
        date = data.index[i]
        previous_date = data.index[i - 1]

        # Check if it's a rebalance date
        if date in rebalance_dates:
            # Use data up to the rebalance date
            historical_data = data[data.index <= date]
            returns = historical_data.pct_change().dropna()
            cov_matrix = returns.cov()

            # Optimize portfolio
            pso = PSO(num_particles=30, num_assets=len(tickers))
            best_pso_allocation = pso.optimize(returns=returns, cov_matrix=cov_matrix, prev_weights=prev_weights)

            # Calculate transaction costs and taxes
            desired_values = best_pso_allocation * portfolio_value
            prices = data.loc[date, tickers]
            desired_shares = desired_values / prices
            shares_diff = desired_shares - np.array([portfolio.shares_held[ticker] for ticker in tickers])

            transaction_costs = transaction_calculator.calculate_total(shares_diff)
            tax_liability = tax_calculator.calculate_total(shares_diff, prices.values, [portfolio.purchase_prices[ticker] for ticker in tickers])

            # Update holdings and purchase prices
            portfolio.update_holdings_after_rebalance(best_pso_allocation, prices, portfolio_value)

            # Update weights
            current_weights = best_pso_allocation
            prev_weights = best_pso_allocation

            # Subtract transaction costs and taxes from portfolio value
            portfolio_value -= (transaction_costs + tax_liability)

        # Calculate daily portfolio return
        daily_asset_returns = data.loc[date, tickers] / data.loc[previous_date, tickers] - 1
        portfolio_return = np.dot(current_weights, daily_asset_returns)
        portfolio_value *= (1 + portfolio_return)
        portfolio_values.loc[date] = portfolio_value

    return portfolio_values