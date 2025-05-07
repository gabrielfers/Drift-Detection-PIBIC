import numpy as np
from river import metrics
from sklearn.metrics import mean_absolute_error


class DriftEvaluator:
    """
    Classe para avaliação e detecção de drift em séries temporais.
    """
    
    @staticmethod
    def inicializar_modelos(modelo_classe, detector_classe):
        """
        Inicializa instâncias de modelo e detector.

        Args:
            modelo_classe: Classe do modelo
            detector_classe: Classe do detector de drift

        Returns:
            tuple: (modelo_instancia, detector_instancia)
        """
        # Instancia o modelo com os parâmetros fornecidos
        modelo_instancia = modelo_classe()
        
        # Instancia o detector com os parâmetros fornecidos
        detector_instancia = detector_classe()

        return modelo_instancia, detector_instancia

    @staticmethod
    def treinamento_modelo_batch(modelo, X, y):
        """
        Treina um modelo em modo batch e calcula erro médio.

        Args:
            modelo: Instância do modelo a ser treinado
            X: Dados de entrada
            y: Valores alvo

        Returns:
            float: Erro médio de treinamento
        """
        # Treinamento do modelo usando o método 'treinar' da subclasse
        modelo.treinar(X, y)

        # Cálculo do erro médio (adapte para modelos online, se necessário)
        erro_medio = mean_absolute_error(y, modelo.prever(X))
      
        return erro_medio
        
    @staticmethod        
    def prequential_batch(X, Y, tamanho_batch, modelo_classe, detector_classe):
        """
        Realiza a previsão de valores continuamente, detectando mudanças nos dados (drift)
        e retreinando o modelo quando necessário.

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch para treinamento inicial e retreinamento.
            modelo_classe: Classe do modelo a ser usado (subclasse de ModeloBase).
            detector_classe: Classe do detector de drift a ser usado (subclasse de DetectorDriftBase).

        Returns:
            predicoes: Lista de previsões.
            deteccoes: Lista de índices onde o drift foi detectado.
        """
        ### variaveis de retorno
        predicoes, erros, deteccoes = [], [], []
        mae = metrics.MAE()


        ### inicializacao do modelo e detector
        modelo, detector = DriftEvaluator.inicializar_modelos(modelo_classe, detector_classe)
        detector.atualizar(DriftEvaluator.treinamento_modelo_batch(modelo, X[:tamanho_batch], Y[:tamanho_batch]))
        
        
        ### variavel de controle
        drift_ativo = False


        ### processamento da stream
        for i in range(tamanho_batch, len(X)):
            
                
            # recebimento do dado de entrada e computacao da previsao
            entrada = X[i].reshape(1, -1)
            y_pred = modelo.prever(entrada)
            erro = mean_absolute_error(Y[i], y_pred)


            # salvando os resultados
            predicoes.append(y_pred)
            erros.append(erro)
            mae.update(Y[i][0], y_pred[0])


            # atualizando o detector
            detector.atualizar(erro)


            # verificando se tem drift
            if detector.drift_detectado:
                deteccoes.append(i)
                #print(f"\nMudança detectada no índice {i}, começando a coletar dados para retreino...")
                drift_ativo = True
                janela_X, janela_y = [], []


            # ativando a estrategia de adaptacao ao drift
            if drift_ativo:
                janela_X.append(X[i])
                janela_y.append(Y[i])

                if len(janela_X) >= tamanho_batch:
                    #print(f"Janela completa com {len(janela_X)} amostras. Retreinado com dados do índice {i - tamanho_batch} até {i}.")
                    drift_ativo = False

                    # realizando o reset do modelo e do detector
                    modelo, detector = DriftEvaluator.inicializar_modelos(modelo_classe, detector_classe)
                    erro_inicial = DriftEvaluator.treinamento_modelo_batch(modelo, np.array(janela_X), np.array(janela_y))
                    detector.atualizar(erro_inicial)

        #print("Modelo utilizado:", modelo)
        #print("Detector utilizado:", detector)
        #print("Detecções:", deteccoes)
        #print(f"MAE Modelo Batch: {mae.get()}")
        
        return [float(p.flatten()[0]) for p in predicoes], deteccoes, mae.get()
    
    @staticmethod
    def prequential_passivo(X, Y, tamanho_batch, modelo_classe):
        """
        Realiza a previsão de valores continuamente para algoritmos online,
        sem detecção de drift e retreinamento.

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch para treinamento inicial.
            modelo_classe: Classe do modelo a ser usado (subclasse de ModeloBase).
            
        Returns:
            predicoes: Lista de previsões.
        """
        
        ### variaveis de retorno
        predicoes, erros = [], []
        mae = metrics.MAE()


        # inicializacao do modelo
        modelo = modelo_classe()
        modelo.treinar(X[:tamanho_batch], Y[:tamanho_batch])
        

        ### processamento da stream
        for i in range(tamanho_batch, len(X)):
            
            
            # recebimento do dado de entrada e computacao da previsao
            y_pred = modelo.prever([X[i]])
            erro = mean_absolute_error(Y[i], np.array([y_pred]))


            # salvando os resultados
            predicoes.append(y_pred)
            mae.update(Y[i][0], y_pred)
            erros.append(erro)


            # computando o aprendizado online
            modelo.treinar([X[i]], [Y[i]])

        
        #print("Modelo utilizado:", modelo)
        #print(f"MAE Modelo Online: {mae.get()}")
        return [float(p.flatten()[0]) for p in np.asarray(predicoes)], mae.get()
        
    @staticmethod
    def prequential_online_com_drift(X, Y, tamanho_batch, modelo_classe, detector_classe):
        """
        Realiza a previsão de valores continuamente para algoritmos online,
        sem detecção de drift e retreinamento.

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch para treinamento inicial.
            modelo_classe: Classe do modelo a ser usado (subclasse de ModeloBase).
            
        Returns:
            predicoes: Lista de previsões.
        """
        
        ### variaveis de retorno
        predicoes, erros, deteccoes = [], [], []
        mae = metrics.MAE()

        # inicializacao do modelo
        modelo, detector = DriftEvaluator.inicializar_modelos(modelo_classe, detector_classe)
        detector.atualizar(DriftEvaluator.treinamento_modelo_batch(modelo, X[:tamanho_batch], Y[:tamanho_batch]))

        ### processamento da stream
        for i in range(tamanho_batch, len(X)):
            
            # recebimento do dado de entrada e computacao da previsao
            y_pred = modelo.prever([X[i]])
            erro = mean_absolute_error(Y[i], np.array([y_pred]))

            # salvando os resultados
            predicoes.append(y_pred)
            mae.update(Y[i][0], y_pred)
            erros.append(erro)
            
            # Atualiza o detector
            detector.atualizar(erro)

            # Verifica detecção de drift
            if detector.drift_detectado:
                deteccoes.append(i)
                detector = detector_classe()

            # computando o aprendizado online
            modelo.treinar([X[i]], [Y[i]])

        
        #print("Modelo utilizado:", modelo)
        #print(f"MAE Modelo Online: {mae.get()}")
        return [float(p.flatten()[0]) for p in np.asarray(predicoes)], deteccoes, mae.get()
    

    @staticmethod
    def prequential_online_com_drift2(X, Y, tamanho_batch, modelo_classe, detector_classe):
        """
        Previsão contínua com modelo online, usando detecção de drift.
        Após detecção de drift, o modelo é atualizado com um batch de dados.
        Fora isso, o modelo apenas prevê (sem atualizar).

        Args:
            X: Dados de entrada.
            Y: Dados de saída.
            tamanho_batch: Tamanho do batch usado para treinamento inicial e após drift.
            modelo_classe: Classe do modelo online com método treinar().
            detector_classe: Classe do detector de drift do River.

        Returns:
            predicoes: Lista de previsões.
            deteccoes: Lista de índices onde drift foi detectado.
            mae_final: Erro médio absoluto.
        """

        predicoes, erros, deteccoes = [], [], []
        mae = metrics.MAE()

        ### inicializacao do modelo e detector
        modelo, detector = DriftEvaluator.inicializar_modelos(modelo_classe, detector_classe)
        detector.atualizar(DriftEvaluator.treinamento_modelo_batch(modelo, X[:tamanho_batch], Y[:tamanho_batch]))

        # Controle de janela de atualização após drift
        atualizando = False
        qtd_dados = 0

        for i in range(tamanho_batch, len(X)):
            x_i = X[i]
            y_i = Y[i][0]

            # Previsão
            y_pred = modelo.prever([x_i])
            erro = mean_absolute_error([y_i], [y_pred])

            # Armazena resultados
            predicoes.append(y_pred)
            erros.append(erro)
            mae.update(y_i, y_pred)

            # Atualiza o detector
            detector.atualizar(erro)

            # Verifica detecção de drift
            if detector.drift_detectado and not atualizando:
                deteccoes.append(i)
                atualizando = True
                detector = detector_classe()

            # Se estiver em modo de atualização, coleta dados
            if atualizando and qtd_dados <= tamanho_batch:
                modelo.treinar(X[i].reshape(1, -1), np.array([[Y[i][0]]]))
                qtd_dados += 1

        return [float(p) for p in predicoes], deteccoes, mae.get()

