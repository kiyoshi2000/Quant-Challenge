import yfinance as yf
import matplotlib.pyplot as plt

def plot_stocks(tickers, start_date, end_date):
    plt.figure(figsize=(12,6))
    for ticker in tickers:
        ticker = ticker.strip()
        # Baixar dados históricos
        data = yf.download(ticker, start=start_date, end=end_date)
        # Verificar se os dados não estão vazios
        if data.empty:
            print(f"Não foram encontrados dados para {ticker}")
            continue
        # Plotar o preço de fechamento
        plt.plot(data.index, data['Close'], label=ticker)
    plt.title('Preço de Fechamento das Ações')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    tickers_input = input("Digite os tickers das ações separados por vírgula (ex: PETR4.SA, VALE3.SA): ")
    tickers = tickers_input.split(',')
    start_date = input("Digite a data de início (AAAA-MM-DD): ")
    end_date = input("Digite a data de término (AAAA-MM-DD): ")
    
    plot_stocks(tickers, start_date, end_date)
