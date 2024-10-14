import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Função para buscar os dados de uma empresa e calcular os principais indicadores trimestrais
def get_company_indicators(ticker):
    # Baixando dados históricos da empresa nos últimos 5 anos
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="5y", interval="3mo")
    
    # Normalizando os índices de datas (removendo fuso horário)
    hist_data.index = hist_data.index.tz_localize(None)
    
    # Aqui você precisaria fornecer o EPS (lucro por ação) e dividendos por trimestre
    # Esses dados normalmente vêm de fontes pagas. Aqui estamos simulando com valores fictícios.
    eps = pd.Series([2, 2.5, 2.8, 3.1, 2.9, 3.2], index=hist_data.index[:6])  # Exemplo de EPS trimestral
    dividends = pd.Series([0.3, 0.3, 0.4, 0.4, 0.35, 0.4], index=hist_data.index[:6])  # Exemplo de dividendos trimestrais

    # Calculando o P/E Ratio manualmente: Preço de fechamento dividido pelo EPS
    pe_ratio = hist_data['Close'] / eps

    # Calculando o Dividend Yield manualmente: Dividendos por ação dividido pelo preço de fechamento
    dividend_yield = dividends / hist_data['Close']

    # Calculando os principais indicadores
    indicators = pd.DataFrame({
        'Date': hist_data.index,
        'Volume': hist_data['Volume'],
        'P/E Ratio': pe_ratio,
        'Dividend Yield': dividend_yield
    })

    return indicators

# As 5 empresas que mais performaram na Bovespa nos últimos 5 anos
top_companies = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA']

# Criando um DataFrame para armazenar os indicadores de todas as empresas
all_indicators = pd.DataFrame()

# Extraindo os indicadores de cada empresa
for company in top_companies:
    company_indicators = get_company_indicators(company)
    company_indicators['Company'] = company
    all_indicators = pd.concat([all_indicators, company_indicators])

# Exibindo a tabela com os dados de Volume, P/E ratio e Dividend Yield
volume_pe_dividend = all_indicators[['Date', 'Company', 'Volume', 'P/E Ratio', 'Dividend Yield']]
print(volume_pe_dividend)

# Plotando os gráficos para Volume, P/E ratio e Dividend Yield
plt.figure(figsize=(12, 8))

# Gráfico de Volume
plt.subplot(3, 1, 1)
for company in top_companies:
    subset = all_indicators[all_indicators['Company'] == company]
    plt.plot(subset['Date'], subset['Volume'], label=f'{company} Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Volume of Top 5 Bovespa Companies (Last 5 Years)')
plt.legend()
plt.grid(True)

# Gráfico de P/E Ratio
plt.subplot(3, 1, 2)
for company in top_companies:
    subset = all_indicators[all_indicators['Company'] == company]
    plt.plot(subset['Date'], subset['P/E Ratio'], label=f'{company} P/E Ratio')
plt.xlabel('Date')
plt.ylabel('P/E Ratio')
plt.title('P/E Ratio of Top 5 Bovespa Companies (Last 5 Years)')
plt.legend()
plt.grid(True)

# Gráfico de Dividend Yield
plt.subplot(3, 1, 3)
for company in top_companies:
    subset = all_indicators[all_indicators['Company'] == company]
    plt.plot(subset['Date'], subset['Dividend Yield'], label=f'{company} Dividend Yield')
plt.xlabel('Date')
plt.ylabel('Dividend Yield')
plt.title('Dividend Yield of Top 5 Bovespa Companies (Last 5 Years)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
