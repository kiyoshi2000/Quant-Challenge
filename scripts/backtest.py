import pandas as pd
import numpy as np
from portfolio import Portfolio
from transaction import TransactionCostCalculator, TaxCalculator
from optimization import PSO
from asset_selection import *
from select_best_assets import *
from dateutil.relativedelta import relativedelta
from collections import defaultdict

def backtest_pso(adj_close_stock, tickers_by_sector_df, adj_close_etf, index_data, start_date, end_date, rebalance_period='1ME', initial_investment=1_000_000, 
                pso_params=None, **kwargs):
    
    portfolio = Portfolio([])
    transaction_calculator = TransactionCostCalculator()
    tax_calculator = TaxCalculator()
    portfolio_allocations = []  # List to store portfolio allocations over time

    portfolio_value = initial_investment
    portfolio_values = pd.Series(index=adj_close_stock.index, dtype=float)
    portfolio_values.iloc[0] = portfolio_value
    # current_weights = np.array([1 / len(tickers)] * len(tickers))  # Start with equal weights
    current_weights = None  # Inicialmente sem alocação
    
    data_full = pd.concat([adj_close_stock, adj_close_etf], axis=1)

    # Generate rebalance dates and ensure they are trading days
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_period)
    rebalance_dates = [date for date in rebalance_dates if date in data_full.index]

    # all_tickers = data_full.columns.tolist()
    # portfolio = Portfolio(all_tickers)
    old_tickers = pd.Series([])

    for i in range(1, len(data_full.index)):
        date = data_full.index[i]
        previous_date = data_full.index[i - 1]

        # Check if it's a rebalance date
        if date in rebalance_dates:
            # Selecionar os tickers atuais
            # selected_tickers = get_selected_tickers(data, date, top_n_per_category=top_n_per_category)
            three_months_before = date - relativedelta(months=3)
            adj_close_stock_backtest = adj_close_stock.loc[three_months_before:date, :]
            adj_close_etf_backtest = adj_close_etf.loc[three_months_before:date, :]

            selected_tickers_stocks = select_best_stocks(adj_close_stock_backtest, index_data, tickers_by_sector_df, market_trending='bull',
                       trading_days=252, risk_free_rate=0.02, metric_to_rank='Sortino_Ratio')
            
            selected_tickers_etfs = select_best_etfs(adj_close_etf_backtest, index_data, market_trending='bull',
                       trading_days=252)
            
            # data = get_data(start_date=start_date, current_date=date, top_n_per_category=top_n_per_category)
            # print(f"Rebalanceamento em {date}: Selecionados {len(selected_tickers)} tickers.")

            selected_tickers = pd.concat([selected_tickers_stocks, selected_tickers_etfs])
            all_tickers = pd.concat([selected_tickers, old_tickers]).unique()

            data = data_full[all_tickers]

            # Atualizar o portfolio com os novos tickers
            num_assets = len(selected_tickers)

            # Use data up to the rebalance date
            historical_data = data.loc[:date,selected_tickers]
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
            desired_values_dict = {k:v for k,v in zip(selected_tickers, desired_values)}

            # Set allocation to 0 for tickers not in the new selection
            for ticker in old_tickers:
                if ticker not in selected_tickers:
                    desired_values_dict[ticker] = 0  # Set value to 0 for removed tickers

            # Combine selected and old tickers
            # all_tickers = list(set(selected_tickers) + set(old_tickers))
            all_tickers = pd.concat([selected_tickers, old_tickers]).unique()

            # Get prices for all tickers
            prices = data.loc[date, ]
            # desired_shares = desired_values / prices # calculating how many assets we should have to achieve the best scenario
            best_allocation_dict = {ticker: desired_values_dict[ticker] / prices[ticker] for ticker in all_tickers}
            prev_allocation_dict = {ticker: portfolio.shares_held.get(ticker, 0) for ticker in all_tickers}

            # Inicializar o dicionário para armazenar as diferenças
            shares_diff_dict = defaultdict(float)

            # Atualizar com as alocações da nova distribuição
            for k, v in best_allocation_dict.items():
                shares_diff_dict[k] += v

            # Subtrair as alocações anteriores
            for k, v in prev_allocation_dict.items():
                shares_diff_dict[k] -= v

            sold_tickers_dict = {k: v for k, v in shares_diff_dict.items() if v < 0}

            # shares_diff = desired_shares - np.array([portfolio.shares_held[ticker] for ticker in selected_tickers]) # how many assets we have to buy/sell to achieve our goal

            transaction_costs = transaction_calculator.calculate_total(shares_diff_dict) # how much does it cost to have this new position
            tax_liability = tax_calculator.calculate_total(sold_tickers_dict, prices, portfolio) # how much taxes we have to pay in case of profit

            # Update holdings and purchase prices
            portfolio.rebalance(shares_diff_dict, prices)

            # Update weights
            current_weights = best_pso_allocation

            # Subtract transaction costs and taxes from portfolio value
            portfolio_value -= (transaction_costs + tax_liability)

            allocation_series = pd.Series(best_pso_allocation, index=selected_tickers, name=date)
            portfolio_allocations.append(allocation_series)

        try:
            old_tickers = selected_tickers
        except:
            old_tickers = pd.Series([])
            
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