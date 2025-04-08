from skopt.space import Real, Integer, Categorical
from .ModelTrainer import ModelTrainer

class Optimizer:
    """
    Classe para definir espaços de parâmetros e chamar o otimizador Bayesiano.
    """

    @staticmethod
    def definir_espacos_parametros():
        """
        Define os espaços de parâmetros para cada modelo

        Retorna:
        --------
        espacos : dict
            Dicionário com espaços de parâmetros para cada modelo
        """
        espacos = {
            # Modelos offline (scikit-learn)
            "LinearRegressionModelo": {
                'fit_intercept': Categorical([True, False]),
                'copy_X': Categorical([True, False]),
                # Remover 'normalize'
                'positive': Categorical([True, False])
            },
            "KNeighborsRegressorModelo": {
                "n_neighbors": Integer(1, 20),
                "weights": Categorical(['uniform', 'distance']),
                "p": Integer(1, 2)
            },
            "LassoRegressionModelo": {
                "alpha": Real(0.001, 10.0, prior='log-uniform'),
                "max_iter": Integer(100, 2000),
                "tol": Real(1e-6, 1e-2, prior='log-uniform')
            },
            "MLPRegressorModelo": {
                "hidden_layer_sizes": Categorical([(50,), (100,), (50, 50), (100, 50)]),
                "activation": Categorical(['relu', 'tanh']),
                "solver": Categorical(['adam', 'sgd']),
                "alpha": Real(1e-5, 1e-2, prior='log-uniform'),
                "learning_rate_init": Real(1e-4, 1e-1, prior='log-uniform')
            },
            "RandomForestModelo": {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(5, 30),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 4)
            },
            "RidgeRegressionModelo": {
                "alpha": Real(0.01, 10.0, prior='log-uniform'),
                "solver": Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'])
            },
            "SVRModelo": {
                "C": Real(0.1, 100, prior='log-uniform'),
                "epsilon": Real(0.01, 1.0, prior='log-uniform'),
                "kernel": Categorical(['linear', 'rbf', 'poly']),
                "gamma": Real(1e-3, 1, prior='log-uniform')
            },

            # Modelos online (River)
            "HoeffdingTreeModelo": {
                "grace_period": Integer(50, 200),
                "split_confidence": Real(1e-10, 1e-1, prior='log-uniform'),
                "leaf_prediction": Categorical(['mean', 'model'])
            },
            "SGDModelo": {
                "loss": Categorical(['squared_error', 'absolute_error']),
                "learning_rate": Real(0.001, 0.1, prior='log-uniform'),
                "l2": Real(0.0, 0.1)
            },
            "LinearRegressionRiverModelo": {
                "optimizer_lr": Real(0.001, 0.1, prior='log-uniform'),
                "l2": Real(0.0, 0.1)
            },
            "PARegressorModelo": {
                "C": Real(0.01, 10.0, prior='log-uniform'),
                "mode": Integer(0, 2)
            },

            # Novos modelos online que estavam faltando
            """
            "BayesianLinearRegressionModelo": {
                "alpha": Real(0.01, 100.0, prior='log-uniform'),
                "beta": Real(0.01, 100.0, prior='log-uniform')
            },
            """

            """
            "HoeffdingTreeRegressorModelo": {
                "grace_period": Integer(50, 200),
                "split_confidence": Real(1e-10, 1e-1, prior='log-uniform'),
                "leaf_prediction": Categorical(['mean', 'model']),
                "max_depth": Integer(5, 50),
                "memory_estimate_period": Integer(1000, 10000)
            },
            """

            """
            "KNNRegressorOnlineModelo": {
                "n_neighbors": Integer(1, 20),
                "window_size": Integer(100, 1000),
                "min_distance": Real(0.0, 1.0)
            },
            """
            "MLPRegressorOnlineModelo": {
                "hidden_dims": Categorical([(10,), (20,), (50,), (10, 10), (20, 10)]),
                "activations": Categorical([("ReLU",), ("Tanh",), ("ReLU", "ReLU"), ("Tanh", "Tanh")]),
                "learning_rate": Real(0.0001, 0.1, prior='log-uniform')
            }
        }

        return espacos

    @staticmethod
    def otimizar_todos_modelos(X, Y, tamanho_batch, lags, optimizer, modelos_offline=None, modelos_online=None):
        """
        Otimiza todos os modelos especificados e retorna os modelos otimizados

        Parâmetros:
        -----------
        X : array-like
            Dados de entrada
        Y : array-like
            Dados alvo
        tamanho_batch : int
            Tamanho do batch para avaliação prequential
        lags : int
            Número de lags na série temporal
        optimizer : OtimizadorBayesiano
            Instância do otimizador Bayesiano
        modelos_offline : list, opcional
            Lista de classes de modelos offline para otimizar
        modelos_online : list, opcional
            Lista de classes de modelos online para otimizar

        Retorna:
        --------
        modelos_otimizados : dict
            Dicionário com modelos otimizados
        parametros_otimizados : dict
            Dicionário com parâmetros otimizados para cada modelo
        """
        espacos = Optimizer.definir_espacos_parametros()

        modelos_otimizados = {}
        parametros_otimizados = {}

        # Otimizar modelos offline (scikit-learn)
        if modelos_offline:
            for modelo_classe in modelos_offline:
                nome_modelo = modelo_classe.__name__
                if nome_modelo in espacos:
                    print(f"\nOtimizando {nome_modelo}...")
                    modelo_otimizado, params = optimizer.otimizar_offline(
                        modelo_classe,
                        espacos[nome_modelo],
                        X, Y
                    )
                    modelos_otimizados[nome_modelo] = modelo_otimizado
                    parametros_otimizados[nome_modelo] = params
                else:
                    print(f"Espaço de parâmetros não definido para {nome_modelo}")

        # Otimizar modelos online (River)
        if modelos_online:
            for modelo_classe in modelos_online:
                nome_modelo = modelo_classe.__name__
                if nome_modelo in espacos:
                    print(f"\nOtimizando {nome_modelo}...")
                    modelo_otimizado, params = optimizer.otimizar_online(
                        modelo_classe,
                        espacos[nome_modelo],
                        X, Y,
                        tamanho_batch
                    )
                    modelos_otimizados[nome_modelo] = modelo_otimizado
                    parametros_otimizados[nome_modelo] = params
                else:
                    print(f"Espaço de parâmetros não definido para {nome_modelo}")

        return modelos_otimizados, parametros_otimizados
