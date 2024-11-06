import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# Função para calcular o retorno esperado do portfólio
def calculate_net_portfolio_return(weights, future_returns, transaction_costs, investment_amount):
    gross_return = np.dot(weights, future_returns)
    net_return = gross_return - (transaction_costs / investment_amount)
    return net_return

# Função para calcular o risco do portfólio
def calculate_portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Função para calcular o Sharpe Ratio
def calculate_sharpe_ratio(portfolio_return, portfolio_risk, risk_free_rate=0.02):
    return (portfolio_return - risk_free_rate) / portfolio_risk

def plot_portfolio_allocations(portfolio_allocations, tickers):
    # Converter as alocações em um DataFrame separado por colunas (uma para cada ativo)
    allocations_df = pd.DataFrame(
        portfolio_allocations['Allocation'].to_list(), 
        columns=tickers, 
        index=portfolio_allocations['Date']
    )

    # Plotar as alocações ao longo do tempo
    plt.figure(figsize=(10, 6))
    
    for ticker in tickers:
        plt.plot(allocations_df.index, allocations_df[ticker], label=ticker)

    plt.title('Pesos do Portfólio ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Alocação (%)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Máximo Drawdown
def calculate_max_drawdown(cumulative_returns):
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_var(returns, confidence_level=0.95):
    mean = returns.mean()
    std_dev = returns.std()
    var = norm.ppf(1 - confidence_level, mean, std_dev)
    return var

# Retornos Anualizados e Sharpe Ratios
def calculate_annualized_return(daily_returns):
    return (1 + daily_returns.mean()) ** 252 - 1

# Sortino Ratios
def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    # Retorno anualizado
    annual_return = calculate_annualized_return(returns)
    # Desvio padrão dos retornos negativos
    downside_returns = returns[returns < 0]
    expected_downside = (downside_returns ** 2).mean()
    downside_deviation = np.sqrt(expected_downside) * np.sqrt(252)
    # Sortino Ratio
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
    return sortino_ratio

def calculate_transaction_costs(weights, prices, shares_traded):
    # Assuming a fixed commission per share
    commission_per_share = 0.005  # $0.005 per share
    min_commission = 1.0          # Minimum $1 per trade

    # Calculate total shares traded per stock
    total_commission = 0
    for i in range(len(weights)):
        shares = shares_traded[i]
        commission = max(commission_per_share * shares, min_commission)
        total_commission += commission

    # Total transaction cost
    total_transaction_cost = total_commission  # In dollars
    return total_transaction_cost

'''def calculate_tax_liability(prev_weights, current_weights, prices, tax_rate, investment_amount = 100_000_000):
    if prev_weights is None:
        return 0  # No taxes on initial investment

    # Calculate gains from selling
    gains = []
    for i in range(len(prev_weights)):
        shares_sold = max(prev_weights[i] - current_weights[i], 0) * investment_amount / prices[i]
        gain = shares_sold * (prices[i] - purchase_prices[i])  # Assuming purchase_prices is known
        gains.append(gain)

    total_gains = sum(gains)
    tax_liability = total_gains * tax_rate
    return tax_liability'''

def adjust_for_inflation(nominal_return, inflation_rate, days_held):
    annualized_inflation = (1 + inflation_rate) ** (days_held / 365) - 1
    real_return = (1 + nominal_return) / (1 + annualized_inflation) - 1
    return real_return

def calculate_management_fees(investment_amount, annual_fee_rate, days_held):
    daily_fee_rate = annual_fee_rate / 252  # Assuming 252 trading days
    fee = investment_amount * daily_fee_rate * days_held
    return fee