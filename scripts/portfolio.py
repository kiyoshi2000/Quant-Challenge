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
        """
        Ajusta as holdings para cada ticker de acordo com as novas alocações.
        Remove posições que não estão mais na nova seleção.
        """
        # Tickers atualmente no portfólio
        current_tickers = set(self.shares_held.keys())
        new_tickers = set(prices.index)

        # Identificar tickers a serem removidos (não estão mais na nova seleção)
        tickers_to_remove = current_tickers - new_tickers

        # Vender todos os tickers que não estão mais na nova seleção
        for ticker in tickers_to_remove:
            shares_change = -self.shares_held[ticker]  # Vender todas as ações
            self.update_holdings(ticker, shares_change, prices[ticker])
            print(f"Vendendo todas as ações de {ticker} durante o rebalanceamento.")

        # Ajustar holdings para os tickers selecionados
        for allocation, ticker in zip(allocations, prices.index):
            desired_value = allocation * portfolio_value
            shares_needed = desired_value / prices[ticker]
            shares_change = shares_needed - self.shares_held.get(ticker, 0)
            self.update_holdings(ticker, shares_change, prices[ticker])