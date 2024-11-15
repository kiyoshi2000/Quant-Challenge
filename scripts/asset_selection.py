# asset_selection.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging 

# ! o s&p não muda muito em um ano. por isso, dá pra rodar no início de cada ano manter aquelas ações. aí, em cada iteração, seleciona as que melhor performarem

# Configuração de Logging
logging.basicConfig(
    filename='../logs/asset_selection.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def adjust_ticker(ticker):
    """
    Ajusta o ticker para ser compatível com o Yahoo Finance.
    Por exemplo, converte 'BRK.B' para 'BRK-B'.
    """
    if '.' in ticker:
        return ticker.replace('.', '-')
    return ticker

def get_sp500_historical(filename='../data/S&P 500 Historical Components & Changes(08-17-2024).csv'):
    """
    Carrega a tabela histórica do S&P 500 a partir de um arquivo CSV.
    
    Retorna:
    - df: DataFrame com as datas como índice e uma coluna 'tickers' contendo listas de tickers.
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col='date', parse_dates=['date'])
        df['tickers'] = df['tickers'].apply(lambda x: sorted([adjust_ticker(ticker.strip()) for ticker in x.split(',')]))
        return df
    else:
        raise FileNotFoundError(f"O arquivo {filename} não foi encontrado.")

def get_sp500_tickers_on_date(df, snapshot_date):
    """
    Obtém a lista de tickers do S&P 500 em uma data específica.
    
    Parâmetros:
    - df: DataFrame histórico do S&P 500.
    - snapshot_date: Data para a qual se deseja obter os tickers.
    
    Retorna:
    - tickers: Lista de tickers presentes no S&P 500 na data especificada.
    """
    df2 = df[df.index <= snapshot_date]
    if df2.empty:
        raise ValueError(f"Não há dados disponíveis para a data {snapshot_date}.")
    
    # Obter a última linha até a snapshot_date
    last_row = df2.tail(1)
    tickers = last_row['tickers'].iloc[0]
    return tickers


def download_tickers_price_data(tickers, end_date, batch_size=100, threads=5):
    """
    Baixa os dados históricos de preços para uma lista de tickers usando yfinance.
    
    Parâmetros:
    - tickers: Lista de tickers a serem baixados.
    - end_date: Data de término no formato 'YYYY-MM-DD'.
    - batch_size: Número de tickers por lote.
    - threads: Número de threads para download paralelo.
    
    Retorna:
    - price_data: DataFrame com os preços ajustados de todos os tickers.
    """
    price_data = pd.DataFrame()
    total_tickers = len(tickers)
    logging.info(f"Iniciando download de preços para {total_tickers} tickers.")

    one_year_before = (end_date - pd.DateOffset(years=1) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    one_year_before = pd.to_datetime(one_year_before)

    for i in range(0, total_tickers, batch_size):
        batch = tickers[i:i+batch_size]
        logging.info(f"Baixando lote {i//batch_size + 1}: {batch}")
        try:
            data = yf.download(batch, start=one_year_before, end=end_date, group_by='ticker', auto_adjust=True, threads=True)
            if len(batch) == 1:
                # Caso haja apenas um ticker no lote, yfinance retorna um DataFrame diferente
                ticker = batch[0]
                data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
            # Reorganizar os dados para um formato padrão
            for ticker in batch:
                if ticker in data.columns.get_level_values(0):
                    ticker_data = data[ticker][['Adj Close','Volume']].rename(ticker)
                    price_data = pd.concat([price_data, ticker_data], axis=1)
                else:
                    logging.warning(f"Dados ausentes para o ticker {ticker}.")
        except Exception as e:
            logging.error(f"Erro ao baixar dados para o lote {batch}: {e}")
    
    price_data.dropna(how='all', inplace=True)  # Remover linhas sem dados
    logging.info("Download de preços concluído.")
    return price_data

def save_price_data(price_data, filename='../data/sp500_historical_prices.csv.gz'):
    """
    Salva os dados de preços ajustados em um arquivo CSV comprimido.
    
    Parâmetros:
    - price_data: DataFrame com os preços ajustados.
    - filename: Nome do arquivo de saída.
    
    Retorna:
    - None
    """
    price_data.to_csv(filename, compression='gzip')
    logging.info(f"Dados de preços salvos em {filename}.")

def load_price_data(filename='data/sp500_historical_prices.csv.gz'):
    """
    Carrega os dados de preços ajustados de um arquivo CSV comprimido.
    
    Parâmetros:
    - filename: Nome do arquivo de entrada.
    
    Retorna:
    - price_data: DataFrame com os preços ajustados.
    """
    if os.path.isfile(filename):
        price_data = pd.read_csv(filename, index_col='Date', parse_dates=True)
        logging.info(f"Dados de preços carregados de {filename}.")
        return price_data
    else:
        raise FileNotFoundError(f"O arquivo {filename} não foi encontrado.")

def categorize_assets(tickers):
    """
    Categoriza os ativos em diferentes buckets.
    Retorna um dicionário com a categoria como chave e a lista de tickers como valor.
    """
    categories = {
        'acoes_tecnologia': [],
        'acoes_farmaceutica': [],
        'acoes_financeiras': [],  # Corrigido para incluir esta categoria
        'renda_fixa_governo': [],
        'renda_fixa_etal_volatilidade_baixa': [],
        'commodities': [],
        'setores_defensivos': [],
        'imoveis': [],
        'infraestrutura': [],
        'protegidos_inflacao': [],
        'moedas_estrangeiras': [],
        'caixa_equivalentes': [],
        'baixa_volatilidade': [],
    }

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            asset_type = info.get('quoteType', 'Unknown')
            
            if asset_type == 'ETF':
                # Categorizar ETFs com base no seu setor ou objetivo
                if 'Technology' in sector:
                    categories['acoes_tecnologia'].append(ticker)
                elif 'Healthcare' in sector:
                    categories['acoes_farmaceutica'].append(ticker)
                elif 'Financial' in sector:
                    categories['acoes_financeiras'].append(ticker)
                elif 'Utilities' in sector:
                    categories['setores_defensivos'].append(ticker)
                elif 'Real Estate' in sector:
                    categories['imoveis'].append(ticker)
                elif 'Infrastructure' in sector:
                    categories['infraestrutura'].append(ticker)
                else:
                    # Categorias específicas de ETFs
                    if 'Renda Fixa' in ticker:
                        categories['renda_fixa_etal_volatilidade_baixa'].append(ticker)
                    elif 'Commodity' in ticker:
                        categories['commodities'].append(ticker)
                    elif 'Inflation' in ticker:
                        categories['protegidos_inflacao'].append(ticker)
                    elif 'Currency' in ticker:
                        categories['moedas_estrangeiras'].append(ticker)
                    elif 'Cash' in ticker or 'BIL' in ticker:
                        categories['caixa_equivalentes'].append(ticker)
                    elif 'Volatility' in ticker:
                        categories['baixa_volatilidade'].append(ticker)
            elif asset_type == 'EQUITY':
                # Categorizar ações individuais com base no setor
                if sector == 'Technology':
                    categories['acoes_tecnologia'].append(ticker)
                elif sector == 'Healthcare':
                    categories['acoes_farmaceutica'].append(ticker)
                elif sector == 'Financials':
                    categories['acoes_financeiras'].append(ticker)
                elif sector in ['Utilities', 'Consumer Staples']:
                    categories['setores_defensivos'].append(ticker)
        except Exception as e:
            print(f"Erro ao obter informações para {ticker}: {e}")
            continue

    return categories

def calculate_beta(returns, benchmark_returns):
    """
    Calcula o beta de um ativo em relação a um benchmark.
    """
    covariance = np.cov(returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    return beta

def calculate_indicators(data, tickers):
    """
    Calcula os indicadores necessários para a seleção de ativos.
    Indicadores: Beta, Retorno, Correlação, etc.
    
    Retorna:
    - indicators: Dicionário com os indicadores para cada ticker.
    """
    indicators = {}
    benchmark_returns = data['^GSPC'].pct_change().dropna()
    for ticker in tickers:
        try:
            returns = data[ticker].pct_change().dropna()
            mean_return = returns.mean()
            beta = calculate_beta(returns, benchmark_returns)
            correlation = returns.corr(benchmark_returns)
            indicators[ticker] = {
                'mean_return': mean_return,
                'beta': beta,
                'correlation': correlation,
            }
        except Exception as e:
            print(f"Erro ao calcular indicadores para {ticker}: {e}")
            continue
    return indicators

def select_top_assets(categories, indicators, top_n=5):
    """
    Seleciona os melhores ativos de cada categoria com base nos indicadores.
    
    Parâmetros:
    - categories: Dicionário de categorias com listas de tickers.
    - indicators: Dicionário de indicadores para cada ticker.
    - top_n: Número de ativos por categoria a serem selecionados.
    
    Retorna:
    - selected_tickers: Lista de tickers selecionados.
    """
    selected_tickers = []
    for category, tickers in categories.items():
        if not tickers:
            continue
        # Selecionar com base no retorno ajustado pelo beta
        sorted_tickers = sorted(
            tickers,
            key=lambda x: indicators[x]['mean_return'] / (indicators[x]['beta'] if indicators[x]['beta'] != 0 else 1),
            reverse=True
        )
        selected = sorted_tickers[:top_n]
        selected_tickers.extend(selected)
    return selected_tickers

def get_selected_tickers(data, current_date, top_n_per_category=5, historical_df=None):
    """
    Função principal para obter os tickers selecionados para o período atual.
    
    Parâmetros:
    - data: DataFrame com os preços históricos ajustados.
    - current_date: Data específica para a seleção de tickers.
    - top_n_per_category: Número de ativos por categoria a serem selecionados.
    - historical_df: DataFrame com a composição histórica do S&P 500.
    
    Retorna:
    - selected_tickers: Lista de tickers selecionados.
    """
    if historical_df is None:
        raise ValueError("O DataFrame histórico do S&P 500 deve ser fornecido.")
    
    # Obter a composição do S&P 500 na data atual
    try:
        tickers = get_sp500_tickers_on_date(historical_df, current_date)
    except Exception as e:
        print(f"Erro ao obter tickers na data {current_date}: {e}")
        return []
    
    if not tickers:
        print(f"Nenhum ticker disponível em {current_date}.")
        return []
    
    # Categorizar os ativos
    categories = categorize_assets(tickers)
    
    # Calcular indicadores
    historical_data = data[data.index <= current_date]
    indicators = calculate_indicators(historical_data, tickers)
    
    # Selecionar os melhores ativos
    selected_tickers = select_top_assets(categories, indicators, top_n=top_n_per_category)
    
    return selected_tickers

def fetch_ticker_info(ticker):
    """
    Função auxiliar para baixar informações detalhadas de um ticker.
    
    Retorna:
    - ticker: O ticker processado.
    - info: Dicionário com as informações do ticker.
    """
    try:
        info = yf.Ticker(ticker).info
        return ticker, {
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown'),
            'Market Cap': info.get('marketCap', np.nan),
            'PE Ratio': info.get('trailingPE', np.nan),
            # Adicione outras informações conforme necessário
        }
    except Exception as e:
        print(f"Erro ao obter informações para {ticker}: {e}")
        return ticker, {
            'Sector': 'Unknown',
            'Industry': 'Unknown',
            'Market Cap': np.nan,
            'PE Ratio': np.nan,
        }

# asset_selection.py (continuação)

def get_data(start_date, snapshot_date, top_n, historical_df, price_data_filename='../data/sp500_historical_prices.csv.gz'):
    """
    Função para obter os dados dos tickers disponíveis no S&P 500 na data especificada,
    otimizados pelo retorno ajustado pelo beta, incluindo informações do yfinance e setor.
    
    Parâmetros:
    - start_date: Data de início para baixar os dados históricos.
    - snapshot_date: Data específica para obter a composição do S&P 500.
    - top_n: Número de ativos por categoria a serem selecionados.
    - historical_df: DataFrame com a composição histórica do S&P 500.
    - price_data_filename: Nome do arquivo CSV comprimido com os dados históricos dos preços.
    
    Retorna:
    - selected_data: DataFrame com os tickers selecionados, suas informações e setor.
    """
    # Carregar os dados de preços históricos
    # Coletar todos os tickers no período
    all_tickers = historical_df['tickers'].explode().unique().tolist()
    # Baixar os dados de preços
    price_data = download_tickers_price_data(
        tickers=all_tickers,
        end_date=snapshot_date,
        batch_size=100,
        threads=5
    )
    
    # Filtrar os dados até a snapshot_date
    price_data = price_data.loc[:snapshot_date]
    
    # Selecionar os tickers
    selected_tickers = get_selected_tickers(
        data=price_data,
        current_date=snapshot_date,
        top_n_per_category=top_n,
        historical_df=historical_df
    )
    
    if not selected_tickers:
        logging.warning(f"Nenhum ticker selecionado para a data {snapshot_date}.")
        return pd.DataFrame()
    
    # Baixar informações detalhadas dos tickers selecionados usando processamento paralelo
    selected_info = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_ticker_info, ticker): ticker for ticker in selected_tickers}
        for future in as_completed(future_to_ticker):
            ticker, info = future.result()
            selected_info[ticker] = info
    
    # Criar DataFrame com as informações
    info_df = pd.DataFrame.from_dict(selected_info, orient='index')
    
    # Obter o último preço ajustado para cada ticker até a snapshot_date
    latest_prices = price_data[selected_tickers].iloc[-1]
    latest_prices.name = 'Adjusted Close'
    
    # Combinar preços e informações
    selected_data = pd.concat([latest_prices, info_df], axis=1)
    
    return selected_data

# asset_selection.py (continuação)

def run_semester_extraction(start_year, end_year, historical_df, price_data_filename='../data/sp500_historical_prices.csv.gz', top_n=5, output_filename='../data/selected_tickers_semesters.csv'):
    """
    Executa a extração de dados para cada semestre entre start_year e end_year,
    seleciona os tickers e salva os resultados em um arquivo CSV.
    
    Parâmetros:
    - start_year: Ano de início.
    - end_year: Ano de término.
    - historical_df: DataFrame com a composição histórica do S&P 500.
    - price_data_filename: Nome do arquivo CSV comprimido com os dados históricos dos preços.
    - top_n: Número de ativos por categoria a serem selecionados.
    - output_filename: Nome do arquivo de saída para salvar os resultados.
    
    Retorna:
    - None
    """
    all_semester_data = []
    iteration = 1
    
    for year in range(start_year, end_year + 1):
        # Definir os semestres
        months = [
            f"{year}-01-01",
            f"{year}-07-01"
        ]
        
        for month_end in months:
            month_end_date = pd.to_datetime(month_end)
            month_start_date = (month_end_date - pd.DateOffset(months=1) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            month_start_date = pd.to_datetime(month_start_date)
            
            print(f"Processando Semestre {iteration}: {month_start_date.strftime('%Y-%m-%d')} a {month_end_date.strftime('%Y-%m-%d')}")
            logging.info(f"Processando Semestre {iteration}: {month_start_date.strftime('%Y-%m-%d')} a {month_end_date.strftime('%Y-%m-%d')}")
            
            # Obter os dados para o semestre atual
            selected_data = get_data(
                start_date=month_start_date.strftime('%Y-%m-%d'),
                snapshot_date=month_start_date,
                top_n=top_n,
                historical_df=historical_df,
                price_data_filename=price_data_filename
            )
            
            if not selected_data.empty:
                selected_data = selected_data.reset_index()
                selected_data.rename(columns={'index': 'Ticker'}, inplace=True)
                selected_data['Iteration'] = iteration
                selected_data['Semester'] = f"{month_end_date.year}-{'H1' if month_end_date.month == 1 else 'H2'}"
                all_semester_data.append(selected_data)
                print(f"Semestre {iteration} processado com sucesso. {len(selected_data)} tickers selecionados.")
                logging.info(f"Semestre {iteration} processado com sucesso. {len(selected_data)} tickers selecionados.")
            else:
                print(f"Semestre {iteration} não teve tickers selecionados.")
                logging.warning(f"Semestre {iteration} não teve tickers selecionados.")
            
            iteration += 1
    
    if all_semester_data:
        # Concatenar todos os DataFrames semestrais
        consolidated_df = pd.concat(all_semester_data, ignore_index=True)
        
        # Salvar o DataFrame consolidado em um arquivo CSV
        consolidated_df.to_csv(output_filename, index=False)
        print(f"Dados consolidados salvos em {output_filename}.")
        logging.info(f"Dados consolidados salvos em {output_filename}.")
    else:
        print("Nenhum dado foi processado.")
        logging.warning("Nenhum dado foi processado.")

def main():
    # Passo 1: Carregar a composição histórica do S&P 500
    historical_df = get_sp500_historical(filename='../data/S&P 500 Historical Components & Changes(08-17-2024).csv')
    print("Composição histórica do S&P 500 carregada com sucesso.")
    
    # Passo 2: Executar a extração semestral e salvar os resultados
    run_semester_extraction(
        start_year=2008,
        end_year=2008,  # Ajuste conforme necessário
        historical_df=historical_df,
        price_data_filename='data/sp500_historical_data.csv.gz',
        top_n=3,
        output_filename='data/selected_tickers_semesters.csv'
    )
    print("Extração semestral concluída e resultados salvos.")
    
    # Passo 3: Executar o backtest utilizando os dados semestrais
    # Carregar os dados históricos de preços
    try:
        historical_data = pd.read_csv('../data/sp500_historical_data.csv.gz', index_col='Date', parse_dates=True)
        print("Dados históricos de preços carregados com sucesso.")
    except FileNotFoundError:
        print("Arquivo 'sp500_historical_data.csv.gz' não encontrado. Por favor, baixe os dados históricos.")
        return
    
    print(historical_data.head())
    
if __name__ == "__main__":
    main()