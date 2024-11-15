# asset_selection.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import time
from tqdm import tqdm  # Para barra de progresso (opcional)
import pyarrow
import wikipedia as wp
import io
import pickle
from datetime import datetime

# Configuração de Logging
logging.basicConfig(
    filename='../logs/asset_selection.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def get_tickers_by_sector(title='List of S&P 500 companies', filename='../data/s&p/sp500.csv', match='Symbol', use_cache=False, filter_tickers=None):
    """
    Fetches a table from Wikipedia and optionally filters it by a list of tickers.

    Parameters:
    - title (str): Title of the Wikipedia page.
    - filename (str): Filename for caching the CSV data.
    - match (str): Text to match to identify the correct table.
    - use_cache (bool): Whether to use the cached file if it exists.
    - filter_tickers (list, optional): List of tickers to filter the DataFrame by.

    Returns:
    - DataFrame: Filtered DataFrame with 'Symbol' and 'GICS Sector' columns.
    """
    # Check if we should use the cached file
    if use_cache and os.path.isfile(filename):
        pass
    else:
        # Fetch the table from Wikipedia
        html = wp.page(title).html()
        df = pd.read_html(io.StringIO(html), header=0, match=match)[0]
        
        # Cache the data to CSV
        df.to_csv(filename, header=True, index=False, encoding='utf-8')
    
    # Load the DataFrame from CSV
    df = pd.read_csv(filename)

    # Select only the 'Symbol' and 'GICS Sector' columns
    df = df[['Symbol', 'GICS Sector']]

    # Filter by the list of tickers if provided
    if filter_tickers is not None:
        df = df[df['Symbol'].isin(filter_tickers)]

    return df

def adjust_ticker(ticker):
    """
    Ajusta o ticker para ser compatível com o Yahoo Finance.
    Por exemplo, converte 'BRK.B' para 'BRK-B'.
    """
    if '.' in ticker:
        return ticker.replace('.', '-')
    return ticker

def get_sp500_tickers_on_date(snapshot_date, data_dir, filename='../data/s&p/S&P 500 Historical Components & Changes(08-17-2024).csv'):
    """
    Carrega a tabela histórica do S&P 500 a partir de um arquivo CSV e salva os tickers no diretório especificado.

    Parâmetros:
    - snapshot_date: Data para a qual obter os tickers do S&P 500 (Timestamp).
    - data_dir: Diretório onde os dados serão salvos.
    - filename: Caminho para o arquivo CSV contendo os componentes históricos do S&P 500.

    Retorna:
    - tickers: Lista de tickers que estavam no S&P 500 na data especificada.
    - tickers_by_sector_df: DataFrame com 'Symbol' e 'GICS Sector' para os tickers.

    Levanta:
    - FileNotFoundError: Se o arquivo CSV não for encontrado.
    - ValueError: Se não houver dados disponíveis para a data especificada.
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col='date', parse_dates=['date'])
        df['tickers'] = df['tickers'].apply(
            lambda x: sorted([adjust_ticker(ticker.strip()) for ticker in x.split(',')])
        )

        # Filtrar as datas até a snapshot_date
        df_before_date = df[df.index <= snapshot_date]
        if df_before_date.empty:
            raise ValueError(f"Não há dados disponíveis para a data {snapshot_date}.")

        # Obter a última linha até a snapshot_date
        last_row = df_before_date.tail(1)
        tickers = last_row['tickers'].iloc[0]

        # Obter tickers por setor
        tickers_by_sector_df = get_tickers_by_sector(filter_tickers=tickers)

        # Salvar os tickers e o DataFrame no diretório específico do período
        tickers_file = os.path.join(data_dir, 'tickers.pkl')
        tickers_by_sector_file = os.path.join(data_dir, 'tickers_by_sector.csv')

        with open(tickers_file, 'wb') as f:
            pickle.dump(tickers, f)

        tickers_by_sector_df.to_csv(tickers_by_sector_file, index=False)

        return tickers, tickers_by_sector_df
    else:
        raise FileNotFoundError(f"O arquivo {filename} não foi encontrado.")
    
def download_single_batch(tickers, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                threads=False,
                progress=False
            )

            if data.empty:
                logging.warning(f"Nenhum dado retornado para os tickers: {tickers}")
                return pd.DataFrame(), pd.DataFrame()

            # Remove informações de fuso horário
            data.index = data.index.tz_localize(None)

            # Verificar se as colunas são MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                # Níveis: [Campo, Ticker]
                if 'Adj Close' in data.columns.levels[0]:
                    adj_close = data['Adj Close']
                else:
                    adj_close = pd.DataFrame()
                    logging.warning(f"'Adj Close' não encontrado nos dados para os tickers: {tickers}")

                if 'Volume' in data.columns.levels[0]:
                    volume = data['Volume']
                else:
                    volume = pd.DataFrame()
                    logging.warning(f"'Volume' não encontrado nos dados para os tickers: {tickers}")
            else:
                # DataFrame de um único ticker
                if 'Adj Close' in data.columns:
                    adj_close = data[['Adj Close']]
                    adj_close.columns = pd.Index([tickers[0]])
                else:
                    adj_close = pd.DataFrame()
                    logging.warning(f"'Adj Close' não encontrado nos dados para o ticker: {tickers[0]}")

                if 'Volume' in data.columns:
                    volume = data[['Volume']]
                    volume.columns = pd.Index([tickers[0]])
                else:
                    volume = pd.DataFrame()
                    logging.warning(f"'Volume' não encontrado nos dados para o ticker: {tickers[0]}")

            return adj_close, volume
        except Exception as e:
            logging.error(f"Erro ao baixar lote de tickers {tickers}: {e}")
            time.sleep(2)  # Esperar antes de tentar novamente

    # Se todas as tentativas falharem, retornar DataFrames vazios
    return pd.DataFrame(), pd.DataFrame()

def download_tickers_price_data(tickers, start_date, end_date, batch_size=100, max_workers=5):
    # Inicializar DataFrames combinados
    combined_adj_close = pd.DataFrame()
    combined_volume = pd.DataFrame()

    # Dividir os tickers em lotes
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submeter todos os lotes para download
        future_to_batch = {
            executor.submit(download_single_batch, batch, start_date, end_date): batch
            for batch in ticker_batches
        }

        for future in tqdm(as_completed(future_to_batch), total=len(ticker_batches), desc="Baixando dados"):
            batch = future_to_batch[future]
            try:
                adj_close_batch, volume_batch = future.result()
                logging.info(f"Processando lote de tickers: {batch}")
                logging.info(f"Shape de adj_close_batch: {adj_close_batch.shape}")
                logging.info(f"Shape de volume_batch: {volume_batch.shape}")
                if not adj_close_batch.empty:
                    combined_adj_close = pd.concat([combined_adj_close, adj_close_batch], axis=1)
                else:
                    logging.warning(f"adj_close_batch vazio para o lote: {batch}")
                if not volume_batch.empty:
                    combined_volume = pd.concat([combined_volume, volume_batch], axis=1)
                else:
                    logging.warning(f"volume_batch vazio para o lote: {batch}")
                logging.info(f"Baixado lote de tickers: {batch}")
            except Exception as e:
                logging.error(f"Erro ao processar lote de tickers {batch}: {e}")

    # Remover colunas completamente vazias
    combined_adj_close.dropna(axis=1, how='all', inplace=True)
    combined_volume.dropna(axis=1, how='all', inplace=True)

    # Verificar se os índices estão alinhados
    if not combined_adj_close.index.equals(combined_volume.index):
        combined_volume = combined_volume.reindex(combined_adj_close.index)

    return combined_adj_close, combined_volume

def filter_valid_tickers(tickers, start_date, end_date):
    valid_tickers = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                logging.warning(f"Ticker sem dados no período: {ticker}")
        except Exception as e:
            logging.warning(f"Erro ao verificar o ticker {ticker}: {e}")
    return valid_tickers

def select_sp_500_tickers(start_date, end_date, data_dir):
    """
    Seleciona os tickers únicos do S&P 500 dentro do período especificado e cria um mapeamento
    das datas de rebalanceamento para os tickers correspondentes.

    Parâmetros:
    - start_date: Data de início para a seleção.
    - end_date: Data de término para a seleção.
    - data_dir: Diretório onde os dados serão salvos.

    Retorna:
    - unique_tickers: Lista de tickers únicos que participaram do S&P 500 durante o período.
    - tickers_per_rebalance_date: Dicionário mapeando datas de rebalanceamento para listas de tickers.
    - tickers_by_sector_df: DataFrame com 'Symbol' e 'GICS Sector' para os tickers do último rebalanceamento.
    """
    # Gerar datas de rebalanceamento trimestral
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QS')  # Início de cada trimestre

    tickers_per_rebalance_date = {}
    all_sp500_tickers = set()
    tickers_by_sector_df = pd.DataFrame()

    for rebalance_date in rebalance_dates:
        try:
            tickers, tickers_by_sector_df = get_sp500_tickers_on_date(rebalance_date, data_dir=data_dir)
            if not tickers:
                logging.warning(f"Nenhum ticker encontrado para a data {rebalance_date}")
            tickers_per_rebalance_date[rebalance_date] = tickers
            all_sp500_tickers.update(tickers)
            logging.info(f"Selecionados {len(tickers)} tickers em {rebalance_date.date()}.")
        except Exception as e:
            logging.error(f"Erro ao obter tickers em {rebalance_date.date()}: {e}")

    unique_tickers = list(all_sp500_tickers)

    # Salvar o mapeamento de tickers por data de rebalanceamento
    tickers_per_rebalance_file = os.path.join(data_dir, 'tickers_per_rebalance_date.pkl')
    with open(tickers_per_rebalance_file, 'wb') as f:
        pickle.dump(tickers_per_rebalance_date, f)

    return unique_tickers, tickers_per_rebalance_date, tickers_by_sector_df

def extract_all_data(start_date, end_date, data_dir=None):
    # Caminhos para os arquivos de dados
    ADJ_CLOSE_DATA_FILE = os.path.join(data_dir, 'adj_close_stock_data.parquet.gzip')
    VOLUME_DATA_FILE = os.path.join(data_dir, 'volume_stock_data.parquet.gzip')

    # Verificar se o arquivo de dados já existe
    if os.path.isfile(ADJ_CLOSE_DATA_FILE):
        logging.info(f"Arquivo de dados encontrado em {ADJ_CLOSE_DATA_FILE}. Carregando dados existentes.")
        adj_close, volume = pd.read_parquet(ADJ_CLOSE_DATA_FILE), pd.read_parquet(VOLUME_DATA_FILE)
        tickers_by_sector_df = pd.read_csv(os.path.join(data_dir, 'tickers_by_sector.csv'))
        with open(os.path.join(data_dir, 'tickers_per_rebalance_date.pkl'), 'rb') as f:
            tickers_per_rebalance_date = pickle.load(f)

    else:
        logging.info("Arquivo de dados não encontrado. Baixando dados.")
        # Baixar todos os dados desde o início
        unique_tickers, tickers_per_rebalance_date, tickers_by_sector_df = select_sp_500_tickers(start_date=start_date, end_date=end_date, data_dir=data_dir)
        
        # Filtrar tickers válidos
        valid_tickers = filter_valid_tickers(unique_tickers, start_date, end_date)
        logging.info(f"{len(valid_tickers)} tickers válidos após filtragem.")

        if valid_tickers:
            adj_close, volume = download_tickers_price_data(
                tickers=unique_tickers,
                start_date=start_date,
                end_date=end_date,
                batch_size=100,
                max_workers=5
            )
            print('data shape: ', adj_close.shape)
            # Salvar os dados baixados
            adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]
            volume = volume.loc[:, ~volume.columns.duplicated()]
            adj_close.to_parquet(ADJ_CLOSE_DATA_FILE, compression='gzip', engine='pyarrow')
            volume.to_parquet(VOLUME_DATA_FILE, compression='gzip', engine='pyarrow')
            logging.info("Dados combinados salvos no arquivo Parquet.")
        else:
            logging.info("Nenhum ticker para baixar.")
            adj_close = pd.DataFrame()
            volume = pd.DataFrame()

    return adj_close, volume, tickers_per_rebalance_date, tickers_by_sector_df

if __name__ == "__main__":
    # Lista de períodos das crises financeiras
    periods = [
        {
            "name": "crise_asiatica",
            "start_date": '1997-07-01',
            "end_date": '1998-12-31'
        },
        {
            "name": "crise_russa",
            "start_date": '1998-01-01',
            "end_date": '1999-12-31'
        },
        {
            "name": "crise_2008",
            "start_date": '2007-01-01',
            "end_date": '2009-12-31'
        },
        {
            "name": "crise_europeia",
            "start_date": '2010-01-01',
            "end_date": '2012-12-31'
        },
        {
            "name": "crise_chinesa",
            "start_date": '2015-06-01',
            "end_date": '2016-12-31'
        },
        {
            "name": "covid",
            "start_date": '2020-01-01',
            "end_date": '2020-12-31'
        },
        {
            "name": "2015-presente",
            "start_date": '2015-01-01',
            "end_date": datetime.today().strftime('%Y-%m-%d')
        }
    ]

    # Diretório base para armazenar os dados
    BASE_DATA_DIR = '../data/stock_data'

    for period in periods:
        period_name = period["name"]
        start_date = period["start_date"]
        end_date = period["end_date"]

        print(f"\nProcessando período: {period_name} ({start_date} a {end_date})")

        # Diretório específico para o período
        DATA_DIR = os.path.join(BASE_DATA_DIR, period_name)
        os.makedirs(DATA_DIR, exist_ok=True)

        # Caminhos para os arquivos Parquet que armazenarão os dados
        ADJ_CLOSE_DATA_FILE = os.path.join(DATA_DIR, f'adj_close_stock_data_{period_name}.parquet.gzip')
        VOLUME_DATA_FILE = os.path.join(DATA_DIR, f'volume_stock_data_{period_name}.parquet.gzip')

        adj_close, volume, tickers_per_rebalance_date, tickers_by_sector_df = extract_all_data(
            start_date=start_date,
            end_date=end_date,
            data_dir=DATA_DIR  # Passar o diretório de dados específico
        )

        # Exibir algumas informações
        print(f"Dados para o período {period_name} salvos.")
        print(f"Shape de adj_close: {adj_close.shape}")
        print(f"Shape de volume: {volume.shape}")

    print("\nProcessamento concluído para todos os períodos.")