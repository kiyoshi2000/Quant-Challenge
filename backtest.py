import pandas as pd
from pso import *
from outils import calculate_net_portfolio_return, calculate_transaction_costs, calculate_management_fees, adjust_for_inflation

# Função de backtest com o PSO
def backtest_pso(data, tickers, start_date, end_date, rebalance_period='3M', initial_investment=100000):
    results = []
    portfolio_allocations = []
    prev_weights = None
    shares_held = {ticker: 0 for ticker in tickers}
    purchase_prices = {ticker: 0.0 for ticker in tickers}
    portfolio_value = initial_investment

    data = data.loc[start_date:end_date]
    data.index = data.index.tz_localize(None)
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_period)

    for date in rebalance_dates:
        # Use data up to the rebalance date
        historical_data = data[data.index <= date]
        returns = historical_data.pct_change().dropna()
        cov_matrix = returns.cov()

        # Optimize portfolio
        pso = PSO(num_particles=30, num_assets=len(tickers))
        best_pso_allocation = pso.optimize(returns=returns, cov_matrix=cov_matrix, prev_weights=prev_weights)

        # Current prices
        prices = data.loc[date, tickers].values

        # Calculate portfolio value (sum of current holdings)
        if prev_weights is not None:
            portfolio_value = sum(shares_held[ticker] * price for ticker, price in zip(tickers, prices))
        else:
            portfolio_value = initial_investment

        # Calculate desired holdings based on new allocation
        desired_values = best_pso_allocation * portfolio_value
        desired_shares = desired_values / prices

        # Calculate changes in holdings
        shares_diff = {}
        for i, ticker in enumerate(tickers):
            shares_diff[ticker] = desired_shares[i] - shares_held[ticker]

        # Initialize variables for transaction costs and tax liability
        transaction_costs = 0.0
        tax_liability = 0.0

        # Process each stock
        for i, ticker in enumerate(tickers):
            shares_change = shares_diff[ticker]
            if shares_change > 0:
                # Buying shares
                # Update the average purchase price
                total_cost_existing = purchase_prices[ticker] * shares_held[ticker]
                total_cost_new = prices[i] * shares_change
                total_shares = shares_held[ticker] + shares_change
                if total_shares > 0:
                    purchase_prices[ticker] = (total_cost_existing + total_cost_new) / total_shares
                else:
                    purchase_prices[ticker] = 0.0
            elif shares_change < 0:
                # Selling shares
                shares_sold = -shares_change
                sale_proceeds = shares_sold * prices[i]
                cost_basis = shares_sold * purchase_prices[ticker]
                capital_gain = sale_proceeds - cost_basis
                # Calculate tax liability (assuming short-term capital gains tax rate of 25%)
                tax_liability += capital_gain * 0.25
                # Update holdings
                if shares_held[ticker] - shares_sold >= 0:
                    shares_held[ticker] -= shares_sold
                else:
                    shares_held[ticker] = 0
            # Update holdings after transactions
            shares_held[ticker] += shares_change

            # Calculate transaction costs (assuming $0.005 per share, min $1 per trade)
            commission = max(0.005 * abs(shares_change), 1.0)
            transaction_costs += commission

        # Calculate net portfolio return
        future_data = data[(data.index > date) & (data.index <= date + pd.DateOffset(months=3))]
        future_returns = future_data.pct_change().dropna()
        # Calculate portfolio return over the period
        period_return = np.dot(best_pso_allocation, future_returns.mean()) * len(future_returns)

        # Update portfolio value
        portfolio_value *= (1 + period_return)
        # Subtract transaction costs and taxes
        portfolio_value -= (transaction_costs + tax_liability)
        # Calculate net period return
        net_period_return = (portfolio_value / initial_investment) - 1

        # Store results
        results.append((date, net_period_return))
        portfolio_allocations.append((date, best_pso_allocation))

        prev_weights = best_pso_allocation

    print("Final Holdings:")
    for ticker in tickers:
        print(f"{ticker}: {shares_held[ticker]:.2f} shares at ${purchase_prices[ticker]:.2f} per share")

    return pd.DataFrame(results, columns=['Date', 'Portfolio Return']), pd.DataFrame(portfolio_allocations, columns=['Date', 'Allocation'])

def backtest_pso_dynamic(data, all_tickers, start_date, end_date, rebalance_period='3M'):
    results = []
    portfolio_allocations = []
    prev_weights = None

    # Rebalance dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_period)

    for date in rebalance_dates:
        # Define the look-back period for stock selection
        lookback_end_date = date - timedelta(days=1)
        lookback_start_date = lookback_end_date - timedelta(days=252)  # 1 year look-back

        # Select stocks based on criteria
        selected_tickers = stock_screener(
            all_tickers, 
            start_date=lookback_start_date.strftime('%Y-%m-%d'), 
            end_date=lookback_end_date.strftime('%Y-%m-%d')
        )

        if not selected_tickers:
            print(f"No stocks selected for date {date}")
            continue

        # Get historical data for selected tickers
        historical_data = yf.download(
            selected_tickers, 
            start=lookback_start_date.strftime('%Y-%m-%d'), 
            end=lookback_end_date.strftime('%Y-%m-%d')
        )['Adj Close']

        # Proceed with PSO optimization as before
        returns = historical_data.pct_change().dropna()
        cov_matrix = returns.cov()

        pso = PSO(num_particles=30, num_assets=len(selected_tickers))
        best_pso_allocation = pso.optimize(returns=returns, cov_matrix=cov_matrix, prev_weights=prev_weights)

        # Store allocations and returns
        portfolio_allocations.append((date, dict(zip(selected_tickers, best_pso_allocation))))

        # Calculate portfolio return for the next period
        future_start_date = date
        future_end_date = date + pd.tseries.offsets.DateOffset(months=3)
        future_data = yf.download(
            selected_tickers, 
            start=future_start_date.strftime('%Y-%m-%d'), 
            end=future_end_date.strftime('%Y-%m-%d')
        )['Adj Close']

        future_returns = future_data.pct_change().dropna().mean()
        portfolio_return = np.dot(best_pso_allocation, future_returns)

        results.append((date, portfolio_return))

        prev_weights = best_pso_allocation

    return pd.DataFrame(results, columns=['Date', 'Portfolio Return']), pd.DataFrame(portfolio_allocations, columns=['Date', 'Allocation'])

def stock_screener(tickers, start_date, end_date):
    selected_stocks = []
    for ticker in tickers:
        try:
            # Get historical price data
            data = yf.download(ticker, start=start_date, end=end_date)
            # Calculate metrics
            price_return = data['Adj Close'][-1] / data['Adj Close'][0] - 1
            volatility = data['Adj Close'].pct_change().std() * np.sqrt(252)
            # Get fundamental data (placeholder for actual data retrieval)
            # e.g., pe_ratio = get_pe_ratio(ticker)
            # Apply your selection criteria
            if price_return > 0.1 and volatility < 0.3:
                selected_stocks.append(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
        print(len(selected_stocks))
    return selected_stocks