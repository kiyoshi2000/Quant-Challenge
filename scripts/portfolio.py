import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, tickers):
        # Initialize holdings and purchase prices for each ticker
        self.shares_held = {ticker: 0 for ticker in tickers}  # Number of shares held for each asset
        self.purchase_prices = {ticker: 0.0 for ticker in tickers}  # Average purchase price for each asset

    def update_holdings(self, ticker, shares_change, current_price):
        """
        Atualiza o número de ações mantidas e o preço médio de compra para um ticker específico.
        """
        if ticker not in self.shares_held:
            self.shares_held[ticker] = 0
            self.purchase_prices[ticker] = 0.0

        if shares_change > 0:
            # Buying more shares: update average purchase price
            total_cost_existing = self.purchase_prices[ticker] * self.shares_held[ticker]  # Total cost of shares currently held
            total_cost_new = current_price * shares_change  # Total cost for the new shares bought
            total_shares = self.shares_held[ticker] + shares_change  # Updated total number of shares
            new_purchase_price = (total_cost_existing + total_cost_new) / total_shares  # New average price
            new_shares_held = total_shares  # Update shares held
        else:
            # Selling shares: no change in average purchase price
            new_shares_held = max(self.shares_held[ticker] + shares_change, 0)  # Ensure shares held don't go negative
            new_purchase_price = self.purchase_prices[ticker]  # Keep the purchase price unchanged

        # Update the holdings and purchase price for the ticker
        self.shares_held[ticker] = new_shares_held
        self.purchase_prices[ticker] = new_purchase_price

    def rebalance(self, shares_diff_dict, prices):
        """
        Ajusta as holdings para cada ticker de acordo com as novas alocações.
        Remove posições que não estão mais na nova seleção.
        """

        # Ajustar holdings para os tickers selecionados
        for ticker, shares in shares_diff_dict.items():
            self.update_holdings(ticker, shares, prices[ticker])