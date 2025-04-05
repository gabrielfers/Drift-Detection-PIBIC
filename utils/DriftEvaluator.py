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

        # Inicializando o modelo e o detector usando as classes
        modelo, detector = ModelTrainer.inicializar_modelos(modelo_classe, detector_classe, **kwargs)

        # Treina o modelo e atualiza o detector
        erro_inicial = ModelTrainer.treinamento_modelo_batch(modelo, X[:tamanho_batch], Y[:tamanho_batch])
        detector.atualizar(erro_inicial)  # Usa o método 'atualizar' da subclasse

        drift_ativo = False

        for i in range(tamanho_batch, len(X)):
            # Realiza a predição usando o método 'prever' da subclasse
            entrada = X[i].reshape(1, -1)
            y_pred = modelo.prever(entrada)[0]
            erro = abs(Y[i][0] - y_pred)

            predicoes.append(y_pred)
            erros.append(erro)
            mae.update(Y[i][0], y_pred)

            # Atualiza o detector usando o método 'atualizar' da subclasse
            detector.atualizar(erro)

            # Se drift for detectado pela primeira vez
            if detector.drift_detectado and not drift_ativo:  # Usa a propriedade 'drift_detectado'
                deteccoes.append(i)
                print(f"\nMudança detectada no índice {i}, começando a coletar dados para retreino...")
                drift_ativo = True
                janela_X, janela_y = [], []

            # Se drift já foi detectado, inicia-se a coleta dos dados
            if drift_ativo:
                janela_X.append(X[i])
                janela_y.append(Y[i])

                if len(janela_X) >= tamanho_batch:
                    print(f"Janela completa com {len(janela_X)} amostras. Retreinado com dados do índice {i - tamanho_batch} até {i}.")
                    drift_ativo = False

                    # Inicializando o modelo e o detector com novas instâncias
                    modelo, detector = ModelTrainer.inicializar_modelos(modelo_classe, detector_classe, **kwargs)

                    # Treina o modelo e atualiza o detector
                    erro_inicial = ModelTrainer.treinamento_modelo_batch(modelo, np.array(janela_X), np.array(janela_y))
                    detector.atualizar(erro_inicial)  # Usa o método 'atualizar' da subclasse

        # Calculando o desvio padrão dos erros com NumPy
        desvio_padrao = np.std(erros)

        print("Modelo utilizado:", modelo)
        print("Detector utilizado:", detector)
        print(f"MAE Modelo Batch: {mae.get()}")
        print(f"Desvio Padrão dos Erros: {desvio_padrao}")

        return predicoes, deteccoes

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
        predicoes = []
        mae = metrics.MAE()

        # Inicializa o modelo usando a classe e kwargs
        modelo = modelo_classe(**kwargs)

        # Treina o modelo com os primeiros exemplos usando treinamento_online_many
        modelo = ModelTrainer.treinamento_online_many(modelo, X, Y, tamanho_batch)

        for i in range(tamanho_batch, len(X)):
            # Converte a entrada para o formato que o modelo online espera
            entrada_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}

            # Realiza a predição usando o método 'prever' da subclasse
            y_pred = modelo.prever([X[i]])  # Passa a entrada como uma lista de uma única amostra

            predicoes.append(y_pred)
            mae.update(Y[i][0], y_pred)

            # Atualiza o modelo online usando o método 'treinar' da subclasse
            modelo.treinar([X[i]], [Y[i]])

        print("Modelo utilizado:", modelo)
        print(f"MAE Modelo Online: {mae.get()}")
        return predicoes
