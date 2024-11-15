from datetime import datetime
import io
import os
import re
import shutil
import numpy as np
import pandas as pd
import wikipedia as wp

def get_table(title='List of S&P 500 companies', filename='sp500.csv', match='Symbol', use_cache=False, filter_tickers=None):
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

filter_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM']  # Example list of tickers

sp500_filtered = get_table(filter_tickers=filter_tickers)

print(sp500_filtered)