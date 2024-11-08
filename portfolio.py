import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, tickers):
        # Initialize holdings and purchase prices for each ticker
        self.shares_held = {ticker: 0 for ticker in tickers}  # Number of shares held for each asset
        self.purchase_prices = {ticker: 0.0 for ticker in tickers}  # Average purchase price for each asset

    def update_holdings(self, ticker, shares_change, current_price):
        # Update the number of shares held and the average purchase price for a specific ticker
        shares_held = self.shares_held[ticker]  # Current number of shares held
        purchase_price = self.purchase_prices[ticker]  # Current average purchase price

        if shares_change > 0:
            # Buying more shares: update average purchase price
            total_cost_existing = purchase_price * shares_held  # Total cost of shares currently held
            total_cost_new = current_price * shares_change  # Total cost for the new shares bought
            total_shares = shares_held + shares_change  # Updated total number of shares
            new_purchase_price = (total_cost_existing + total_cost_new) / total_shares  # New average price
            new_shares_held = total_shares  # Update shares held
        else:
            # Selling shares: no change in average purchase price
            new_shares_held = max(shares_held + shares_change, 0)  # Ensure shares held don't go negative
            new_purchase_price = purchase_price  # Keep the purchase price unchanged

        # Update the holdings and purchase price for the ticker
        self.shares_held[ticker] = new_shares_held
        self.purchase_prices[ticker] = new_purchase_price

    def rebalance(self, allocations, prices, portfolio_value):
        # Adjust holdings for each ticker according to new target allocations
        for allocation, ticker in zip(allocations, prices.index):
            # price = prices[]
            # Calculate desired value for each asset based on target allocation
            desired_value = allocation * portfolio_value
            # Determine the number of shares needed to match the desired value
            shares_needed = desired_value / prices[ticker]
            # Calculate the difference in shares needed to achieve target allocation
            shares_change = shares_needed - self.shares_held[ticker]
            # Update holdings based on calculated share difference
            self.update_holdings(ticker, shares_change, prices[ticker])