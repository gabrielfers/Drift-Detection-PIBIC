import numpy as np
from river import metrics
from .ModelTrainer import ModelTrainer

class DriftEvaluator:
    """
    Classe para avaliação e detecção de drift em séries temporais.
    """

    @staticmethod
    def prequential_batch(X, Y, tamanho_batch, modelo_classe, detector_classe, **kwargs):
        """
        Realiza a previsão de valores continuamente, detectando mudanças nos dados (drift)
        e retreinando o modelo quando necessário.

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch para treinamento inicial e retreinamento.
            modelo_classe: Classe do modelo a ser usado (subclasse de ModeloBase).
            detector_classe: Classe do detector de drift a ser usado (subclasse de DetectorDriftBase).
            **kwargs: Parâmetros adicionais para o modelo e detector.

        Returns:
            predicoes: Lista de previsões.
            deteccoes: Lista de índices onde o drift foi detectado.
        """
        predicoes, erros, deteccoes = [], [], []
        mae = metrics.MAE()

        modelo, detector = ModelTrainer.inicializar_modelos(modelo_classe, detector_classe, **kwargs)

        erro_inicial = ModelTrainer.treinamento_modelo_batch(modelo, X[:tamanho_batch], Y[:tamanho_batch])
        detector.atualizar(erro_inicial)

        drift_ativo = False

        for i in range(tamanho_batch, len(X)):
            entrada = X[i].reshape(1, -1)
            y_pred = modelo.prever(entrada)[0]
            erro = abs(Y[i][0] - y_pred)

            predicoes.append(y_pred)
            erros.append(erro)
            mae.update(Y[i][0], y_pred)

            detector.atualizar(erro)

            if detector.drift_detectado and not drift_ativo:
                deteccoes.append(i)
                print(f"\nMudança detectada no índice {i}, começando a coletar dados para retreino...")
                drift_ativo = True
                janela_X, janela_y = [], []

            if drift_ativo:
                janela_X.append(X[i])
                janela_y.append(Y[i])

                if len(janela_X) >= tamanho_batch:
                    print(f"Janela completa com {len(janela_X)} amostras. Retreinado com dados do índice {i - tamanho_batch} até {i}.")
                    drift_ativo = False

                    modelo, detector = ModelTrainer.inicializar_modelos(modelo_classe, detector_classe, **kwargs)

                    erro_inicial = ModelTrainer.treinamento_modelo_batch(modelo, np.array(janela_X), np.array(janela_y))
                    detector.atualizar(erro_inicial)

        desvio_padrao = np.std(erros)

        print("Modelo utilizado:", modelo)
        print("Detector utilizado:", detector)
        print(f"MAE Modelo Batch: {mae.get()}")
        print(f"Desvio Padrão dos Erros: {desvio_padrao}")

        return predicoes, deteccoes, mae, desvio_padrao

    @staticmethod
    def prequential_passivo(X, Y, tamanho_batch, modelo_classe, **kwargs):
        """
        Realiza a previsão de valores continuamente para algoritmos online,
        sem detecção de drift e retreinamento.

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch para treinamento inicial.
            modelo_classe: Classe do modelo a ser usado (subclasse de ModeloBase).
            **kwargs: Parâmetros adicionais para o modelo.

        Returns:
            predicoes: Lista de previsões.
        """
        predicoes, erros = [], []
        mae = metrics.MAE()

        modelo = modelo_classe(**kwargs)

        modelo = ModelTrainer.treinamento_online_many(modelo, X, Y, tamanho_batch)

        for i in range(tamanho_batch, len(X)):
            entrada_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}

            y_pred = modelo.prever([X[i]])
            erro = abs(Y[i][0] - y_pred)

            predicoes.append(y_pred)
            mae.update(Y[i][0], y_pred)
            erros.append(erro)

            modelo.treinar([X[i]], [Y[i]])

        desvio_padrao = np.std(erros)
        print("Modelo utilizado:", modelo)
        print(f"MAE Modelo Online: {mae.get()}")
        return predicoes, mae, desvio_padrao