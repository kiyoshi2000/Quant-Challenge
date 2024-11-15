import pandas as pd
import numpy as np
from portfolio import Portfolio
from transaction import TransactionCostCalculator, TaxCalculator
from optimization import PSO
from asset_selection import *

def backtest_pso(data, start_date, end_date, rebalance_period='1ME', initial_investment=1_000_000, 
                pso_params=None, top_n_per_category=5, **kwargs):
    # portfolio = Portfolio(tickers)
    transaction_calculator = TransactionCostCalculator()
    tax_calculator = TaxCalculator()
    portfolio_allocations = []  # List to store portfolio allocations over time

    portfolio_value = initial_investment
    portfolio_values = pd.Series(index=data.index, dtype=float)
    portfolio_values.iloc[0] = portfolio_value
    # current_weights = np.array([1 / len(tickers)] * len(tickers))  # Start with equal weights
    current_weights = None  # Inicialmente sem alocação

    # Generate rebalance dates and ensure they are trading days
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_period)
    rebalance_dates = [date for date in rebalance_dates if date in data.index]

    portfolio = Portfolio([])  # Inicialmente vazio

    for i in range(1, len(data.index)):
        date = data.index[i]
        previous_date = data.index[i - 1]

        # Check if it's a rebalance date
        if date in rebalance_dates:
            # Selecionar os tickers atuais
            selected_tickers = get_selected_tickers(data, date, top_n_per_category=top_n_per_category)
            data = get_data(start_date=start_date, current_date=date, top_n_per_category=top_n_per_category)
            print(f"Rebalanceamento em {date}: Selecionados {len(selected_tickers)} tickers.")

            # Atualizar o portfolio com os novos tickers
            num_assets = len(selected_tickers)

            # Use data up to the rebalance date
            historical_data = data[data.index <= date]
            returns = historical_data.pct_change().dropna()
            cov_matrix = returns.cov()

            # Optimize portfolio
            pso = PSO(
                num_assets=num_assets,
                **(pso_params if pso_params else {})
            )

            # Inicializar pesos anteriores ou igual ponderado se for o primeiro rebalanceamento
            if current_weights is not None and len(current_weights) == num_assets:
                prev_weights = current_weights
            else:
                prev_weights = np.array([1/num_assets]*num_assets) if num_assets > 0 else np.array([])

            best_pso_allocation = pso.optimize(
                returns=returns,
                cov_matrix=cov_matrix,
                prev_weights=prev_weights,
                portfolio_value=portfolio_value,
                portfolio_values=portfolio_values,
                **kwargs
            )

            # Calculate transaction costs and taxes
            desired_values = best_pso_allocation * portfolio_value # how much money will be allocated to each asset
            prices = data.loc[date, selected_tickers]
            desired_shares = desired_values / prices # calculating how many assets we should have to achieve the best scenario
            shares_diff = desired_shares - np.array([portfolio.shares_held[ticker] for ticker in selected_tickers]) # how many assets we have to buy/sell to achieve our goal

            transaction_costs = transaction_calculator.calculate_total(shares_diff) # how much does it cost to have this new position
            tax_liability = tax_calculator.calculate_total(shares_diff, prices.values, [portfolio.purchase_prices[ticker] for ticker in selected_tickers]) # how much taxes we have to pay in case of profit

            # Update holdings and purchase prices
            portfolio.rebalance(best_pso_allocation, prices, portfolio_value)

            # Update weights
            current_weights = best_pso_allocation

            # Subtract transaction costs and taxes from portfolio value
            portfolio_value -= (transaction_costs + tax_liability)

            allocation_series = pd.Series(best_pso_allocation, index=selected_tickers, name=date)
            portfolio_allocations.append(allocation_series)
            print(date)

        # Calcular retorno diário do portfólio
        if portfolio.shares_held:
            try:
                # Filtrar tickers com holdings > 0 para evitar erros
                active_tickers = [ticker for ticker, shares in portfolio.shares_held.items() if shares > 0]
                if active_tickers:
                    daily_asset_returns = data.loc[date, active_tickers] / data.loc[previous_date, active_tickers] - 1
                    weights_array = np.array([current_weights[portfolio_allocations[-1].index.get_loc(ticker)] 
                                              for ticker in active_tickers if ticker in portfolio_allocations[-1].index])
                    portfolio_return = np.dot(weights_array, daily_asset_returns[active_tickers])
                    portfolio_value *= (1 + portfolio_return)
                portfolio_values.loc[date] = portfolio_value
            except Exception as e:
                print(f"Erro ao calcular retorno diário em {date.date()}: {e}")
                portfolio_values.loc[date] = portfolio_value
        else:
            portfolio_values.loc[date] = portfolio_value

    # After the loop, create a DataFrame from the allocations
    allocations_df = pd.DataFrame(portfolio_allocations)

    # Set the index to the rebalance dates
    allocations_df.index = pd.to_datetime(allocations_df.index)

    return portfolio_values, allocations_df