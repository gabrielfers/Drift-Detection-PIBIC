# Arquivo: otimizador_bayesiano_prequential.py
import time
from skopt import BayesSearchCV, gp_minimize
from skopt.utils import use_named_args
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
import warnings
from river import metrics
from copy import deepcopy

class OtimizadorBayesiano:
    """
    Classe para realizar busca bayesiana de hiperparâmetros para modelos online (River) e offline (scikit-learn).
    """
    
    def __init__(self, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=1):
        """
        Inicializa o otimizador bayesiano
        
        Parâmetros:
        -----------
        n_iter : int
            Número de iterações para a busca bayesiana
        cv : int
            Número de folds para validação cruzada (apenas modelos offline)
        random_state : int
            Seed para reprodutibilidade
        n_jobs : int
            Número de jobs paralelos (-1 usa todos os processadores)
        verbose : int
            Nível de detalhamento das saídas
        """
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def otimizar_offline(self, modelo_classe, espaco_parametros, X, y):
        """
        Otimiza hiperparâmetros para modelos offline (scikit-learn)
        
        Parâmetros:
        -----------
        modelo_classe : classe do modelo
            Classe do modelo scikit-learn a ser otimizado
        espaco_parametros : dict
            Dicionário com espaço de busca para cada hiperparâmetro
        X : array-like
            Dados de entrada
        y : array-like
            Dados alvo
            
        Retorna:
        --------
        modelo_otimizado : objeto modelo
            Modelo com os melhores hiperparâmetros
        melhores_params : dict
            Dicionário com os melhores hiperparâmetros encontrados
        """
        print(f"Iniciando otimização bayesiana para {modelo_classe.__name__}")
        
        # Criando o objeto de busca bayesiana
        tempo_inicio = time.time()
        
        # Instanciando modelo base
        modelo_base = modelo_classe().get_model() if hasattr(modelo_classe, 'get_model') else modelo_classe()
        
        # Configurando busca bayesiana
        optimizer = BayesSearchCV(
            modelo_base,
            espaco_parametros,
            n_iter=self.n_iter,
            cv=self.cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Executando a busca
        optimizer.fit(X, y)
        
        # Obtendo resultados
        tempo_execucao = time.time() - tempo_inicio
        
        print(f"Otimização concluída em {tempo_execucao:.2f} segundos")
        print(f"Melhores parâmetros: {optimizer.best_params_}")
        print(f"Melhor score: {optimizer.best_score_:.4f}")
        
        # Criando modelo otimizado
        if hasattr(modelo_classe, 'get_model'):
            modelo_otimizado = modelo_classe(**optimizer.best_params_)
        else:
            modelo_otimizado = modelo_classe().set_params(**optimizer.best_params_)
            
        return modelo_otimizado, optimizer.best_params_
    
    def otimizar_online(self, modelo_classe, espaco_parametros, X, y, tamanho_batch):
        """
        Otimiza hiperparâmetros para modelos online (River)
        
        Parâmetros:
        -----------
        modelo_classe : classe do modelo
            Classe do modelo River a ser otimizado
        espaco_parametros : dict
            Dicionário com espaço de busca para cada hiperparâmetro
        X : array-like
            Dados de entrada
        y : array-like
            Dados alvo
        tamanho_batch : int
            Tamanho do batch para avaliação prequential
            
        Retorna:
        --------
        modelo_otimizado : objeto modelo
            Modelo com os melhores hiperparâmetros
        melhores_params : dict
            Dicionário com os melhores hiperparâmetros encontrados
        """
        print(f"Iniciando otimização bayesiana para modelo online {modelo_classe.__name__}")
        
        # Converter espaço de parâmetros para formato de lista para iteração
        param_grid = {}
        for param_name, param_space in espaco_parametros.items():
            if isinstance(param_space, Real):
                param_grid[param_name] = np.linspace(param_space.low, param_space.high, 10)
            elif isinstance(param_space, Integer):
                param_grid[param_name] = np.arange(param_space.low, param_space.high + 1)
            elif isinstance(param_space, Categorical):
                param_grid[param_name] = param_space.categories
                
        # Função para avaliar um conjunto de parâmetros
        def avaliar_configuracao(params):
            modelo = modelo_classe(**params)
            mae_acumulado = self._validar_modelo_online(modelo, X, y, tamanho_batch)
            return mae_acumulado
            
        # Usar BayesSearchCV adaptado para modelos online
        from skopt import gp_minimize
        
        # Preparando o espaço de busca para o otimizador bayesiano
        dimensions = list(espaco_parametros.values())
        param_names = list(espaco_parametros.keys())
        
        # Função objetivo para minimização
        def objetivo(values):
            # Converter valores para dicionário de parâmetros
            params = {param_names[i]: values[i] for i in range(len(param_names))}
            return avaliar_configuracao(params)
            
        tempo_inicio = time.time()
        
        # Executando a otimização bayesiana
        resultado = gp_minimize(
            func=objetivo,
            dimensions=dimensions,
            n_calls=self.n_iter,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Convertendo resultados para dicionário
        melhores_params = {param_names[i]: resultado.x[i] for i in range(len(param_names))}
        
        tempo_execucao = time.time() - tempo_inicio
        
        print(f"Otimização concluída em {tempo_execucao:.2f} segundos")
        print(f"Melhores parâmetros: {melhores_params}")
        print(f"Melhor score: {resultado.fun:.4f}")
        
        # Criando modelo otimizado
        modelo_otimizado = modelo_classe(**melhores_params)
            
        return modelo_otimizado, melhores_params
    
    def _validar_modelo_online(self, modelo, X, y, tamanho_batch):
        """
        Avalia um modelo online usando avaliação prequential em batches
        
        Retorna:
        --------
        erro_medio : float
            Erro médio (MAE) do modelo
        """
        # Métricas para avaliar o modelo
        erro_metrica = metrics.MAE()
        
        n_samples = len(y)
        predicoes = []
        
        # Avaliação prequential em batches
        for i in range(0, n_samples, tamanho_batch):
            end_idx = min(i + tamanho_batch, n_samples)
            batch_X = X[i:end_idx]
            batch_y = y[i:end_idx]
            
            # Previsão do batch atual
            batch_predicoes = []
            for j in range(len(batch_X)):
                # Para modelos online, fazemos predict antes do learn
                pred = modelo.predict_one(batch_X[j])
                batch_predicoes.append(pred)
                # Atualiza o modelo
                modelo.learn_one(batch_X[j], batch_y[j])
                # Atualiza a métrica
                erro_metrica.update(batch_y[j], pred)
            
            predicoes.extend(batch_predicoes)
            
        return erro_metrica.get()
