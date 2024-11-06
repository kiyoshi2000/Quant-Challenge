import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, tickers):
        self.shares_held = {ticker: 0 for ticker in tickers}
        self.purchase_prices = {ticker: 0.0 for ticker in tickers}

    def update_holdings(self, ticker, shares_change, current_price):
        shares_held = self.shares_held[ticker]
        purchase_price = self.purchase_prices[ticker]

        if shares_change > 0:
            # Buying shares
            total_cost_existing = purchase_price * shares_held
            total_cost_new = current_price * shares_change
            total_shares = shares_held + shares_change
            new_purchase_price = (total_cost_existing + total_cost_new) / total_shares if total_shares > 0 else 0.0
            new_shares_held = total_shares
        elif shares_change < 0:
            # Selling shares
            new_shares_held = max(shares_held + shares_change, 0)  # Ensure shares held don't go negative
            new_purchase_price = purchase_price  # Purchase price remains the same when selling
        else:
            # No change in shares
            new_shares_held = shares_held
            new_purchase_price = purchase_price

        self.shares_held[ticker] = new_shares_held
        self.purchase_prices[ticker] = new_purchase_price

    def update_holdings_after_rebalance(self, current_weights, prices, portfolio_value):
        desired_values = current_weights * portfolio_value
        desired_shares = desired_values / prices

        shares_diff = desired_shares - np.array([self.shares_held[ticker] for ticker in prices.index])
        shares_diff = pd.Series(shares_diff, index=prices.index)

        for ticker in prices.index:
            shares_change = shares_diff[ticker]
            current_price = prices[ticker]
            self.update_holdings(ticker, shares_change, current_price)
