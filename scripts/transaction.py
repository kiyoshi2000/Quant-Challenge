import numpy as np

class TaxCalculator:
    def __init__(self, tax_rate=0.25):
        self.tax_rate = tax_rate

    def calculate(self, shares_sold, sale_price, purchase_price):
        sale_proceeds = shares_sold * sale_price
        cost_basis = shares_sold * purchase_price
        capital_gain = sale_proceeds - cost_basis
        tax_liability = capital_gain * self.tax_rate if capital_gain > 0 else 0
        return tax_liability

    def old_calculate_total(self, shares_diffs, prices, purchase_prices):
        total_tax = 0.0
        for i, shares_change in enumerate(shares_diffs):
            if shares_change < 0:
                shares_sold = -shares_change
                current_price = prices[i]
                purchase_price = purchase_prices[i]
                total_tax += self.calculate(shares_sold, current_price, purchase_price)
        return total_tax

    def calculate_total(self, shares_diffs, prices, portfolio):
        total_tax = 0.0
        for ticker, shares_change in shares_diffs.items():
            shares_sold = -shares_change
            current_price = prices[ticker]
            purchase_price = portfolio.purchase_prices[ticker]
            total_tax += self.calculate(shares_sold, current_price, purchase_price)
        return total_tax
    
class TransactionCostCalculator:
    def __init__(self, commission_per_share=0.005, min_commission=1.0):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def calculate(self, shares_change):
        commission = max(self.commission_per_share * abs(shares_change), self.min_commission)
        return commission

    def old_calculate_total(self, shares_diffs):
        total_cost = 0.0
        for shares_change in shares_diffs:
            total_cost += self.calculate(shares_change)
        return total_cost

    def calculate_total(self, shares_diffs):
        total_cost = 0.0
        for _, shares_change in shares_diffs.items():
            total_cost += self.calculate(shares_change)
        return total_cost