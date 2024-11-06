import pandas as pd
from pso import *
from outils import calculate_transaction_costs, calculate_portfolio_return, update_holdings, calculate_capital_gains_tax
from pandas.tseries.offsets import BDay

# Backtesting function with PSO optimization
def backtest_pso(data, tickers, start_date, end_date, rebalance_period='3ME', initial_investment=100000):
    """
    Perform backtesting of the portfolio using PSO optimization.

    Parameters:
    - data: DataFrame of historical adjusted closing prices.
    - tickers: List of stock tickers.
    - start_date: Start date of the backtest.
    - end_date: End date of the backtest.
    - rebalance_period: Frequency of rebalancing (e.g., '3M' for every 3 months).
    - initial_investment: Initial amount of money invested.

    Returns:
    - results_df: DataFrame containing dates and portfolio returns.
    - allocations_df: DataFrame containing dates and portfolio allocations.
    """
    results = []  # List to store portfolio returns over time
    portfolio_allocations = []  # List to store portfolio allocations over time
    prev_weights = None  # Previous portfolio weights
    shares_held = {ticker: 0 for ticker in tickers}  # Number of shares held for each stock
    purchase_prices = {ticker: 0.0 for ticker in tickers}  # Purchase price per share for each stock
    portfolio_value = initial_investment  # Total value of the portfolio
    cumulative_returns = []  # List to store cumulative returns

    # Filter data for the backtest period and remove timezone information
    data = data.loc[start_date:end_date]

    # Generate rebalance dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_period)

    for date in rebalance_dates:

        # Find the next business day
        next_trading_day = date
        next_trading_day = next_trading_day.tz_localize(None)

        # Ensure the next trading day is in your data
        while next_trading_day not in data.index and next_trading_day <= max(data.index):
            next_trading_day += BDay(1)
            next_trading_day = next_trading_day.tz_localize(None)

        if next_trading_day > max(data.index): break
            
        # Use historical data up to the rebalance date
        historical_data = data[data.index <= next_trading_day]
        # Calculate daily returns
        returns = historical_data.pct_change().dropna()
        # Calculate covariance matrix of returns
        cov_matrix = returns.cov()

        # Optimize portfolio using PSO
        pso = PSO(num_particles=30, num_assets=len(tickers))
        best_pso_allocation = pso.optimize(returns=returns, cov_matrix=cov_matrix, prev_weights=prev_weights)            

        # Current stock prices at rebalance date
        prices = data.loc[next_trading_day, tickers].values

        # Calculate current portfolio value based on existing holdings
        if prev_weights is not None:
            portfolio_value = sum(shares_held[ticker] * price for ticker, price in zip(tickers, prices))
        else:
            portfolio_value = initial_investment

        # Desired dollar value and number of shares for each stock based on new allocation
        desired_values = best_pso_allocation * portfolio_value
        desired_shares = desired_values / prices

        # Calculate net period return
        net_period_return = (portfolio_value / initial_investment) - 1

        # Store cumulative return
        cumulative_returns.append((date, portfolio_value))

        # Calculate changes in holdings (number of shares to buy or sell)
        shares_diff = {}
        for i, ticker in enumerate(tickers):
            shares_diff[ticker] = desired_shares[i] - shares_held[ticker]

        # Initialize total transaction costs and tax liability for this rebalance
        transaction_costs = 0.0
        tax_liability = 0.0

        # Process each stock for buying or selling
        for i, ticker in enumerate(tickers):
            shares_change = shares_diff[ticker]  # Number of shares to buy (>0) or sell (<0)
            current_price = prices[i]            # Current price per share
            purchase_price = purchase_prices[ticker]  # Average purchase price per share
            shares_current = shares_held[ticker]      # Current number of shares held

            # Update holdings and purchase price
            new_shares_held, new_purchase_price = update_holdings(
                shares_current, purchase_price, shares_change, current_price
            )
            shares_held[ticker] = new_shares_held
            purchase_prices[ticker] = new_purchase_price

            # Calculate transaction costs for this trade
            transaction_costs += calculate_transaction_costs(shares_change)

            # Calculate tax liability if selling shares
            if shares_change < 0:
                shares_sold = -shares_change  # Number of shares sold
                tax_liability += calculate_capital_gains_tax(
                    shares_sold, current_price, purchase_price
                )

        # Calculate portfolio return over the next period
        future_data = data[(data.index > next_trading_day) & (data.index <= next_trading_day + pd.DateOffset(months=3))]
        future_returns = future_data.pct_change().dropna()

        # Handle cases where future returns might be empty
        if future_returns.empty:
            print(f"No future returns data available after {next_trading_day}. Ending backtest.")
            break

        # Calculate the portfolio return over the period
        period_return = calculate_portfolio_return(best_pso_allocation, future_returns)

        # Update portfolio value with the period return
        portfolio_value *= (1 + period_return)
        # Subtract transaction costs and tax liabilities from portfolio value
        portfolio_value -= (transaction_costs + tax_liability)
        # Calculate net period return relative to initial investment
        net_period_return = (portfolio_value / initial_investment) - 1

        # Store the results
        results.append((next_trading_day, net_period_return))
        portfolio_allocations.append((next_trading_day, best_pso_allocation))

        # Update previous weights for next iteration
        prev_weights = best_pso_allocation
    
    # Return cumulative returns as a DataFrame
    cumulative_returns_df = pd.DataFrame(cumulative_returns, columns=['Date', 'Portfolio Value'])
    cumulative_returns_df.set_index('Date', inplace=True)

    # Create DataFrames from results
    results_df = pd.DataFrame(results, columns=['Date', 'Portfolio Return'])
    allocations_df = pd.DataFrame(portfolio_allocations, columns=['Date', 'Allocation'])

    return results_df, allocations_df, cumulative_returns_df

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