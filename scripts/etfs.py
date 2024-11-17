import yfinance as yf
import os
import pandas as pd
from datetime import datetime

def baixar_dados(tickers, start_date, end_date, data_dir, prefix):
    """
    Baixa os dados de preços ajustados e volumes dos tickers fornecidos
    e os salva no diretório especificado.

    Args:
        tickers (list): Lista de tickers.
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.
        data_dir (str): Diretório onde os arquivos serão salvos.
        prefix (str): Prefixo para os nomes dos arquivos salvos.
    """
    # Criar diretório, se não existir
    os.makedirs(data_dir, exist_ok=True)

    # Baixar dados
    print(f"Baixando dados para {prefix}...")
    adj_close = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    volume = yf.download(tickers, start=start_date, end=end_date)['Volume']

    # Salvar dados em arquivos Parquet
    adj_close_path = os.path.join(data_dir, f"{prefix}_adj_close.parquet")
    volume_path = os.path.join(data_dir, f"{prefix}_volume.parquet")

    adj_close.to_parquet(adj_close_path, compression='gzip', engine='pyarrow')
    volume.to_parquet(volume_path, compression='gzip', engine='pyarrow')

    print(f"Dados salvos para {prefix}:")
    print(f"- {adj_close_path}")
    print(f"- {volume_path}")

def main():
    """
    Processa os períodos de crise, baixa os dados e salva os resultados.
    """
    # Diretório de destino
    DATA_DIR = '../data/etfs'

    # Configurações dos períodos de crise
    datasets = [
        {
            "tickers": ['GC=F', '^TNX', '^TYX'],
            "start_date": '1996-07-01',
            "end_date": '1998-12-31',
            "prefix": 'crise_asiatica',
        },
        {
            "tickers": ['CHF=X', 'JPY=X', '^TNX', '^TYX'],
            "start_date": '1996-01-01',
            "end_date": '1999-12-31',
            "prefix": 'crise_russa',
        },
        {
            "tickers": ['TLT', 'GLD', 'XLP'],
            "start_date": '2006-01-01',
            "end_date": '2009-12-31',
            "prefix": 'crise_2008',
        },
        {
            "tickers": ['GLD', 'IEF', 'FXF'],
            "start_date": '2009-01-01',
            "end_date": '2012-12-31',
            "prefix": 'crise_europeia',
        },
        {
            "tickers": ['GLD', 'TLT', 'XLU'],
            "start_date": '2014-06-01',
            "end_date": '2016-12-31',
            "prefix": 'crise_chinesa',
        },
        {
            "tickers": ['BND', 'GLD', 'XLU'],
            "start_date": '2019-01-01',
            "end_date": '2020-12-31',
            "prefix": 'covid',
        },
        {
            "tickers": ['TLT', 'GLD', 'IAU', 'TIP', 'XLP', 'XLU', 'XLV', 'FXY', 'FXF'],
            "start_date": '2014-01-01',
            "end_date": '2024-11-11',
            "prefix": '2015-presente',
        }
    ]

    # Processar cada conjunto de dados
    for dataset in datasets:
        tickers = dataset["tickers"]
        start_date = dataset["start_date"]
        end_date = dataset["end_date"]
        prefix = dataset["prefix"]

        try:
            baixar_dados(tickers, start_date, end_date, DATA_DIR, prefix)
        except Exception as e:
            print(f"Erro ao processar {prefix}: {e}")

if __name__ == "__main__":
    main()