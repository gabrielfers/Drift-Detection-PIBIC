from collections import deque
from fastdtw import fastdtw
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import copy 

class SmartWindow:
    def __init__(self, tam_janela=50, max_janelas=3, limiar_drift=0.5, limiar_dtw=30, modelo=None):
        """
        Inicializa a SmartWindow.

        Args:
            tam_janela (int): Número de pontos por janela.
            max_janelas (int): Número máximo de janelas guardadas.
            limiar_drift (float): MAE mínimo para sinalizar drift.
            limiar_dtw (float): Distância DTW máxima para manter janela.
            modelo: Modelo de regressão (ou outro) a ser usado. Usa LinearRegression se None.
                     Uma cópia profunda do modelo será criada.
        """
        self.TAM_JANELA = tam_janela
        self.MAX_JANELAS = max_janelas
        self.LIMIAR_DRIFT = limiar_drift
        self.LIMIAR_DTW = limiar_dtw

        self.buffer = deque(maxlen=self.TAM_JANELA)
        self.janelas = []  # lista de tuplas: (DataFrame, vetor_erro)

        if modelo is not None:
            self._modelo_inicial = copy.deepcopy(modelo)
            self.modelo = copy.deepcopy(modelo)
        else:
            # Se nenhum modelo for fornecido, cria um novo LinearRegression
            self._modelo_inicial = LinearRegression()
            self.modelo = LinearRegression()

    def _extrair_xy(self, df):
        """Retorna X (features) e y (target) de um DataFrame."""
        # Assume que a última coluna é 'y' e as anteriores são features 'x'
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    def _calcular_erro(self, y_true, y_pred):
        """Retorna o vetor de erro absoluto."""
        return np.abs(y_true - y_pred)

    def _detectar_drift(self, erro):
        """Retorna True se o MAE ultrapassar LIMIAR_DRIFT."""
        mae = np.mean(erro)
        return mae > self.LIMIAR_DRIFT

    def _calcular_dtw(self, e_antigo, e_novo):
        """Calcula e retorna a distância DTW entre dois vetores de erro."""
        distance, _ = fastdtw(e_antigo, e_novo)
        return distance

    def _train_model(self, X, y):
        """Treina o modelo com os dados fornecidos (assume interface scikit-learn)."""
        if hasattr(self.modelo, 'fit'):
            # Assume interface scikit-learn para modelos offline/batch
            self.modelo.fit(X, y)
        else:
            raise AttributeError("O modelo fornecido não possui um método 'fit' reconhecido.")

    def _predict_batch(self, X):
        """Faz predições em batch (assume interface scikit-learn)."""
        if hasattr(self.modelo, 'predict'):
             return self.modelo.predict(X) # Para sklearn ou outros
        else:
            raise AttributeError("O modelo fornecido não possui um método 'predict' reconhecido.")

    def processar_ponto(self, x_i, y_i):
        """
        Processa um novo ponto (x_i, y_i) do stream.

        Args:
            x_i: Features do novo ponto (pode ser um escalar ou array).
            y_i: Target do novo ponto.

        Returns:
            bool: True se um drift foi detectado e o modelo foi retreinado, False caso contrário.
        """
        # 1) Adiciona o novo ponto ao buffer
        # Garante que x_i seja sempre um iterável (lista ou array)
        if not hasattr(x_i, '__iter__'):
            x_i = [x_i]
        ponto_data = list(x_i) + [y_i]
        self.buffer.append(ponto_data)

        # 2) Aguarda até encher a janela
        if len(self.buffer) < self.TAM_JANELA:
            return False

        # Constrói DataFrame da janela atual
        colunas = [f'x{j}' for j in range(len(x_i))] + ['y']
        window_df = pd.DataFrame(list(self.buffer), columns=colunas)
        Xn, yn = self._extrair_xy(window_df)

        # 3) Se for a primeira janela, faz o treinamento inicial
        if not self.janelas:
            self._train_model(Xn, yn)
            y_pred_inicial = self._predict_batch(Xn)
            erro_inicial = self._calcular_erro(yn, y_pred_inicial)
            self.janelas.append((window_df.copy(), erro_inicial))
            return False

        # Calcula predição e erro para a janela atual usando o modelo atual (antes do retreino)
        y_pred = self._predict_batch(Xn)
        erro_novo = self._calcular_erro(yn, y_pred)

        drift_detectado = False
        # 4) Detecta drift pelo vetor de erro
        if self._detectar_drift(erro_novo):
            drift_detectado = True
            # 5) Filtra janelas antigas via DTW entre vetores de erro
            janelas_validas_dfs = []
            for old_df, erro_antigo in self.janelas:
                dist_dtw = self._calcular_dtw(erro_antigo, erro_novo)
                if dist_dtw < self.LIMIAR_DTW:
                    janelas_validas_dfs.append(old_df)

            # 6) Re-treina o modelo com janelas válidas + a nova
            if janelas_validas_dfs:
                combinado = pd.concat(janelas_validas_dfs + [window_df], ignore_index=True)
            else:
                combinado = window_df # Se nenhuma janela antiga for válida, treina só com a nova

            Xc, yc = self._extrair_xy(combinado)
            self._train_model(Xc, yc) # Retreina o modelo

            # Recalcula o erro da janela atual com o modelo retreinado para armazenar
            y_pred_retreinado = self._predict_batch(Xn)
            erro_novo = self._calcular_erro(yn, y_pred_retreinado) # Atualiza o erro com o modelo novo

        # 7) Atualiza lista de janelas (FIFO)
        self.janelas.append((window_df.copy(), erro_novo))
        if len(self.janelas) > self.MAX_JANELAS:
            self.janelas.pop(0)

        return drift_detectado

    def reset(self):
        """
        Reseta o estado da SmartWindow para o estado inicial.
        Limpa o buffer, as janelas armazenadas e reinstancia o modelo
        usando uma cópia profunda do modelo inicial.
        """
        self.buffer.clear()
        self.janelas = []
        # Reinstancia o modelo usando uma cópia profunda do modelo inicial armazenado
        self.modelo = copy.deepcopy(self._modelo_inicial)
