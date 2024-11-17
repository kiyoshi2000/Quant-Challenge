from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import yfinance as yf
import numpy as np

def combine_data(adj_close, index_data):
    """
    Combina os dados de preços ajustados das ações com os dados do índice.

    Args:
        adj_close (pd.DataFrame): Preços ajustados das ações.
        index_data (pd.Series): Dados do índice.

    Returns:
        pd.DataFrame: DataFrame combinado com ações e índice.
    """
    combined_data = adj_close.join(index_data, how='inner')
    combined_data = combined_data.dropna(axis=1, how='any')
    return combined_data

def calculate_daily_returns(combined_data):
    """
    Calcula os retornos diários a partir dos dados combinados.

    Args:
        combined_data (pd.DataFrame): DataFrame combinado com ações e índice.

    Returns:
        pd.DataFrame: Retornos diários das ações e do índice.
    """
    returns = combined_data.pct_change(fill_method=None).dropna()
    return returns

def calculate_annualized_returns(returns, trading_days=252):
    """
    Calcula os retornos anualizados a partir dos retornos diários.

    Args:
        returns (pd.DataFrame): Retornos diários.
        trading_days (int): Número de dias de negociação por ano.

    Returns:
        pd.Series: Retornos anualizados.
    """
    mean_daily_return = returns.mean()
    annualized_return = (1 + mean_daily_return) ** trading_days - 1
    return annualized_return

def calculate_sortino_ratio(returns, annualized_return, trading_days=252, risk_free_rate=0.02):
    """
    Calcula o Índice de Sortino para cada ação.

    Args:
        returns (pd.DataFrame): Retornos diários.
        annualized_return (pd.Series): Retornos anualizados.
        trading_days (int): Número de dias de negociação por ano.
        risk_free_rate (float): Taxa livre de risco anual.

    Returns:
        pd.Series: Índice de Sortino de cada ação.
    """
    # Retorno anualizado excedente
    excess_annual_return = annualized_return - risk_free_rate

    # Calcular o desvio padrão dos retornos negativos
    negative_returns = returns.copy()
    negative_returns[negative_returns > 0] = np.nan
    downside_deviation = negative_returns.std()
    annualized_downside_deviation = downside_deviation * np.sqrt(trading_days)
    annualized_downside_deviation = annualized_downside_deviation.replace(0, np.nan)

    # Índice de Sortino
    sortino_ratio = excess_annual_return / annualized_downside_deviation
    return sortino_ratio

def calculate_beta(returns):
    """
    Calcula o Beta de cada ação em relação ao índice.

    Args:
        returns (pd.DataFrame): Retornos diários.

    Returns:
        pd.Series: Beta de cada ação.
    """
    covariance = returns.cov()
    index_variance = returns['Index'].var()
    beta = covariance.loc[:, 'Index'] / index_variance
    beta = beta.drop('Index', errors='ignore')
    return beta

def filter_by_market_trend(metrics_df, market_trending):
    """
    Filtra as ações com base na tendência do mercado.

    Args:
        metrics_df (pd.DataFrame): DataFrame com as métricas calculadas.
        market_trending (str): Tendência do mercado ('bull' ou 'bear').

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    if market_trending == 'bull':
        metrics_df_filtered = metrics_df.loc[metrics_df['Beta'] > 0, :]
    else:
        metrics_df_filtered = metrics_df.loc[metrics_df['Beta'] < 0, :]
    return metrics_df_filtered

def select_top_stocks(metrics_df_filtered, tickers_by_sector_df, metric_to_rank='Sortino_Ratio'):
    """
    Seleciona as top 5 ações em cada setor com base em uma métrica.

    Args:
        metrics_df_filtered (pd.DataFrame): DataFrame filtrado pelas métricas.
        tickers_by_sector_df (pd.DataFrame): DataFrame com os setores das ações.
        metric_to_rank (str): Métrica para ordenação.

    Returns:
        pd.DataFrame: DataFrame com as top ações por setor.
    """
    # Combinar com o DataFrame dos setores
    df_final = pd.merge(tickers_by_sector_df, metrics_df_filtered, on='Symbol', how='inner')

    # Selecionar as 5 melhores ações em cada setor
    top_stocks_per_sector = df_final.groupby('GICS Sector').apply(
        lambda x: x.nlargest(3, metric_to_rank)
    ).reset_index(drop=True)
    return top_stocks_per_sector

def select_best_stocks(adj_close, index_data, tickers_by_sector_df, market_trending='bull',
                       trading_days=252, risk_free_rate=0.02, metric_to_rank='Sortino_Ratio'):
    """
    Função principal que seleciona as melhores ações com base nas métricas.

    Args:
        adj_close (pd.DataFrame): Preços ajustados das ações.
        index_data (pd.Series): Dados do índice.
        tickers_by_sector_df (pd.DataFrame): DataFrame com os setores das ações.
        market_trending (str): Tendência do mercado ('bull' ou 'bear').
        trading_days (int): Número de dias de negociação por ano.
        risk_free_rate (float): Taxa livre de risco anual.
        metric_to_rank (str): Métrica para ordenação.

    Returns:
        tuple: DataFrame com as top ações por setor e DataFrame com as métricas calculadas.
    """
    # Etapa 1: Preparação dos Dados
    combined_data = combine_data(adj_close, index_data)
    returns = calculate_daily_returns(combined_data)

    # Etapa 2: Cálculo das Métricas
    annualized_return = calculate_annualized_returns(returns, trading_days)
    sortino_ratio = calculate_sortino_ratio(returns, annualized_return, trading_days, risk_free_rate)
    beta = calculate_beta(returns)

    # Criar o DataFrame com as métricas
    metrics_df = pd.DataFrame({
        'Symbol': annualized_return.index.drop('Index'),
        'Annualized_Return': annualized_return.drop('Index').values,
        'Sortino_Ratio': sortino_ratio.drop('Index').values,
        'Beta': beta.values
    })

    # Remover linhas com NaN nas métricas
    metrics_df = metrics_df.dropna()

    # Etapa 3: Filtragem com Base na Tendência do Mercado
    metrics_df_filtered = filter_by_market_trend(metrics_df, market_trending)

    # Etapa 4: Seleção das Melhores Ações
    top_stocks_per_sector = select_top_stocks(metrics_df_filtered, tickers_by_sector_df, metric_to_rank)

    return top_stocks_per_sector.Symbol

def select_best_etfs(adj_close, index_data, market_trending='bull',
                       trading_days=252):
    """
    Função principal que seleciona as melhores ações com base nas métricas.

    Args:
        adj_close (pd.DataFrame): Preços ajustados das ações.
        index_data (pd.Series): Dados do índice.
        tickers_by_sector_df (pd.DataFrame): DataFrame com os setores das ações.
        market_trending (str): Tendência do mercado ('bull' ou 'bear').
        trading_days (int): Número de dias de negociação por ano.
        risk_free_rate (float): Taxa livre de risco anual.
        metric_to_rank (str): Métrica para ordenação.

    Returns:
        tuple: DataFrame com as top ações por setor e DataFrame com as métricas calculadas.
    """
    # Etapa 1: Preparação dos Dados
    combined_data = combine_data(adj_close, index_data)
    returns = calculate_daily_returns(combined_data)

    # Etapa 2: Cálculo das Métricas
    annualized_return = calculate_annualized_returns(returns, trading_days)
    beta = calculate_beta(returns)

    # Criar o DataFrame com as métricas
    metrics_df = pd.DataFrame({
        'Symbol': annualized_return.index.drop('Index'),
        'Annualized_Return': annualized_return.drop('Index').values,
        'Beta': beta.values
    })

    # Remover linhas com NaN nas métricas
    metrics_df = metrics_df.dropna()

    # Etapa 3: Filtragem com Base na Tendência do Mercado
    metrics_df_filtered = filter_by_market_trend(metrics_df, market_trending)

    return metrics_df_filtered.Symbol

if __name__ == '__main__':
    period_name = 'presente'
    BASE_STOCK_DIR = '../data/stock_data'
    BASE_ETF_DIR = '../data/etfs'
    DATA_DIR = os.path.join(BASE_STOCK_DIR, period_name)
    ADJ_CLOSE_STOCK_FILE = os.path.join(DATA_DIR, 'adj_close_stock_data.parquet.gzip')
    ADJ_CLOSE_ETF_FILE = os.path.join(BASE_ETF_DIR, '2015-presente_adj_close.parquet')

    adj_close_stock = pd.read_parquet(ADJ_CLOSE_STOCK_FILE)
    adj_close_etf = pd.read_parquet(ADJ_CLOSE_ETF_FILE)
    adj_close_etf.index = adj_close_etf.index.tz_localize(None)

    tickers_by_sector_df = pd.read_csv(os.path.join(DATA_DIR, 'tickers_by_sector.csv'))
    with open(os.path.join(DATA_DIR, 'tickers_per_rebalance_date.pkl'), 'rb') as f:
        tickers_per_rebalance_date = pickle.load(f)

    index_symbol = '^GSPC'  # S&P 500
    start_date = adj_close_stock.index.min().strftime('%Y-%m-%d')
    end_date = adj_close_stock.index.max().strftime('%Y-%m-%d')

    # Baixar os dados do índice
    index_data = yf.download(index_symbol, start=start_date, end=end_date)['Adj Close']
    index_data.name = 'Index'

    # Chamar a função principal
    df = select_best_stocks(adj_close_stock, index_data, tickers_by_sector_df, market_trending='bull')
    df2 = select_best_etfs(adj_close_etf, index_data, market_trending='bull')
    print(df)
    print(df2)