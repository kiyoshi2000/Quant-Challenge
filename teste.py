import multiprocessing

def worker(num):
    """Função que simula uma tarefa."""
    print(f"Processo {num} está executando")

if __name__ == "__main__":
    processes = []
    for i in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()