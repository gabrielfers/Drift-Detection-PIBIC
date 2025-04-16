import pandas as pd

class SaveCSV:
    def __init__(self, filename):
        self.filename = filename

    def salvar_resultados_csv(dados_para_csv, nome_arquivo_csv):
        df_resultados = pd.DataFrame(dados_para_csv)

        nome_arquivo_csv = f'{nome_arquivo_csv}.csv'
        df_resultados.to_csv(nome_arquivo_csv, index=False)

        print(f"\nResultados salvos em {nome_arquivo_csv}")

        print("\nResumo dos Resultados:")
        print(df_resultados)