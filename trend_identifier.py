# trend_identifier.py

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

class MarketTrendIdentifier:
    def __init__(self, 
                 tickers, 
                 benchmark_ticker='^GSPC', 
                 start_date='2015-01-01', 
                 end_date='2024-04-30',
                 trend_window=252,  # 1 ano
                 beta_window=126,   # ~6 meses
                 ma_short=50,
                 ma_long=200):
        """
        Inicializa o identificador de tendência de mercado.
        """
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.trend_window = trend_window
        self.beta_window = beta_window
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.data = None
        self.benchmark_data = None
        self.load_data()

    def load_data(self):
        """
        Carrega os dados históricos dos ativos e do benchmark.
        """
        print("Baixando dados históricos...")
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        self.benchmark_data = yf.download(self.benchmark_ticker, start=self.start_date, end=self.end_date)['Adj Close']
        # Remover informações de timezone
        self.data.index = self.data.index.tz_localize(None)
        self.benchmark_data.index = self.benchmark_data.index.tz_localize(None)
        print("Dados carregados com sucesso.")

    def calculate_beta(self):
        """
        Calcula o beta de cada ativo em relação ao benchmark.
        """
        print("Calculando beta para cada ativo...")
        returns = self.data.pct_change().dropna()
        benchmark_returns = self.benchmark_data.pct_change().dropna()
        # Renomear a série do benchmark para garantir o nome correto na concatenação
        benchmark_returns = benchmark_returns.rename(self.benchmark_ticker)
        beta_values = {}

        for ticker in self.tickers:
            # Alinhar datas
            if ticker not in returns.columns:
                print(f"Aviso: Ticker {ticker} não encontrado nos dados retornados.")
                continue
            df = pd.concat([returns[ticker], benchmark_returns], axis=1).dropna()
            asset_returns = df[ticker]
            try:
                market_returns = df[self.benchmark_ticker]
            except KeyError:
                print(f"Erro: Coluna '{self.benchmark_ticker}' não encontrada no DataFrame.")
                continue
            slope, intercept, r_value, p_value, std_err = linregress(market_returns, asset_returns)
            beta_values[ticker] = slope

        beta_df = pd.DataFrame.from_dict(beta_values, orient='index', columns=['Beta'])
        print("Beta calculado com sucesso.")
        return beta_df

    def identify_market_trend(self):
        """
        Identifica a tendência atual do mercado usando médias móveis.
        """
        print("Identificando a tendência do mercado...")
        ma_short = self.benchmark_data.rolling(window=self.ma_short).mean()
        ma_long = self.benchmark_data.rolling(window=self.ma_long).mean()
        latest_ma_short = ma_short.iloc[-1]
        latest_ma_long = ma_long.iloc[-1]
        prev_ma_short = ma_short.iloc[-2]
        prev_ma_long = ma_long.iloc[-2]

        if latest_ma_short > latest_ma_long and prev_ma_short <= prev_ma_long:
            trend = 'subindo'
        elif latest_ma_short < latest_ma_long and prev_ma_short >= prev_ma_long:
            trend = 'descendo'
        else:
            # Determinar se está lateralizando baseado na inclinação
            recent_prices = self.benchmark_data[-self.trend_window:]
            slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(recent_prices)), recent_prices.values)
            if abs(slope) < 0.0001:
                trend = 'lateralizando'
            elif slope > 0:
                trend = 'subindo'
            else:
                trend = 'descendo'

        print(f"Tendência atual do mercado: {trend}")
        return trend

    def adjust_asset_weights(self, beta_df, trend):
        """
        Ajusta os pesos dos ativos com base na tendência do mercado e no beta.
        """
        print("Ajustando os pesos dos ativos com base na tendência do mercado...")
        if trend == 'subindo':
            # Priorizar ativos com beta alto e positivo
            positive_betas = beta_df[beta_df['Beta'] > 0].sort_values(by='Beta', ascending=False)
            if positive_betas.empty:
                weights = pd.Series(1 / len(beta_df), index=beta_df.index)
            else:
                weights = positive_betas['Beta'] / positive_betas['Beta'].sum()
                # Para ativos com beta <=0, set weight to 0
                zero_betas = beta_df[beta_df['Beta'] <= 0]
                for ticker in zero_betas.index:
                    weights[ticker] = 0
        elif trend == 'descendo':
            # Priorizar ativos com beta baixo e negativo
            negative_betas = beta_df[beta_df['Beta'] < 0].sort_values(by='Beta', ascending=True)
            if negative_betas.empty:
                weights = pd.Series(1 / len(beta_df), index=beta_df.index)
            else:
                weights = negative_betas['Beta'].abs() / negative_betas['Beta'].abs().sum()
                # Para ativos com beta >=0, set weight to 0
                zero_betas = beta_df[beta_df['Beta'] >= 0]
                for ticker in zero_betas.index:
                    weights[ticker] = 0
        else:
            # Lateralizando: distribuição equilibrada
            weights = pd.Series(1 / len(beta_df), index=beta_df.index)

        # Garantir que todos os pesos sejam não-negativos
        weights = weights.clip(lower=0)
        # Normalizar para somar 1
        if weights.sum() != 0:
            weights /= weights.sum()
        else:
            weights = pd.Series(1 / len(beta_df), index=beta_df.index)

        print("Pesos ajustados com sucesso:")
        print(weights)
        return weights

    def dynamic_parameter_adjustment(self, trend):
        """
        Ajusta dinamicamente os parâmetros de otimização com base na tendência do mercado.
        """
        print("Ajustando parâmetros de otimização dinamicamente...")
        if trend == 'subindo':
            # Priorizar retorno, tolerar mais risco
            params = {
                'w_return': 0.4,
                'w_momentum': 0.2,
                'w_deviation': -0.1,
                'w_sortino': 0.2,
                'w_portfolio_volatility': -0.2,
                'w_cvar': -1.0,
                'w_drawdown': -0.5,
                'w_concentration': -0.3,
                'w_transaction_cost': -0.1
            }
        elif trend == 'descendo':
            # Priorizar redução de risco
            params = {
                'w_return': 0.2,
                'w_momentum': 0.1,
                'w_deviation': -0.2,
                'w_sortino': 0.3,
                'w_portfolio_volatility': -0.3,
                'w_cvar': -1.5,
                'w_drawdown': -0.7,
                'w_concentration': -0.4,
                'w_transaction_cost': -0.2
            }
        else:
            # Lateralizando: equilíbrio entre risco e retorno
            params = {
                'w_return': 0.3,
                'w_momentum': 0.15,
                'w_deviation': -0.15,
                'w_sortino': 0.25,
                'w_portfolio_volatility': -0.25,
                'w_cvar': -1.2,
                'w_drawdown': -0.6,
                'w_concentration': -0.35,
                'w_transaction_cost': -0.15
            }
        print("Parâmetros ajustados:")
        for key, value in params.items():
            print(f"{key}: {value}")
        return params

    def visualize_trend(self):
        """
        Visualiza as médias móveis e a tendência atual do mercado.
        """
        print("Visualizando a tendência do mercado...")
        plt.figure(figsize=(14,7))
        plt.plot(self.benchmark_data, label='Preço do Benchmark')
        ma_short = self.benchmark_data.rolling(window=self.ma_short).mean()
        ma_long = self.benchmark_data.rolling(window=self.ma_long).mean()
        plt.plot(ma_short, label=f'MA {self.ma_short} dias')
        plt.plot(ma_long, label=f'MA {self.ma_long} dias')
        plt.title('Tendência do Mercado com Médias Móveis')
        plt.xlabel('Data')
        plt.ylabel('Preço Ajustado')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """
        Executa todo o processo de identificação de tendência, cálculo de beta, ajuste de pesos e parâmetros.
        """
        trend = self.identify_market_trend()
        beta_df = self.calculate_beta()
        weights = self.adjust_asset_weights(beta_df, trend)
        params = self.dynamic_parameter_adjustment(trend)
        self.visualize_trend()
        return weights, params, trend

if __name__ == "__main__":
    # Exemplo de uso
    tickers = [
        'NVDA', 'BRK-B', 'C', 'JPM', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'JNJ',
        'TLT', 'GLD', 'EFA', 'EEM', 'LQD'
    ]

    trend_identifier = MarketTrendIdentifier(
        tickers=tickers,
        benchmark_ticker='^GSPC',
        start_date='2015-01-01',
        end_date='2024-04-30',
        trend_window=252,
        beta_window=126,
        ma_short=50,
        ma_long=200
    )

    weights, params, trend = trend_identifier.run()

    # Salvar os resultados em arquivos, se necessário
    weights.to_csv('C:\\Users\\Edison Araki\\OneDrive\\Área de Trabalho\\Project\\Quant-Challenge\\adjusted_weights.csv')   
    pd.Series(params).to_csv('C:\\Users\\Edison Araki\\OneDrive\\Área de Trabalho\\Project\\Quant-Challenge\\adjusted_params.csv')

 