import numpy as np

class PerformanceMetrics:
    @staticmethod
    def calculate_annualized_return(returns):
        cumulative_return = (1 + returns).prod()
        num_periods = len(returns)
        annualized_return = cumulative_return ** (252 / num_periods) - 1
        return annualized_return

    @staticmethod
    def calculate_annualized_volatility(returns):
        return returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        excess_returns = returns - (risk_free_rate / 252)
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        cumulative_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / cumulative_max - 1
        max_drawdown = drawdown.min()
        return max_drawdown