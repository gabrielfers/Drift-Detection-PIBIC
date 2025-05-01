from sklearn.neighbors import KNeighborsRegressor
from regressores.ModeloBase import ModeloBase
from utils.FileManager import FileManager


class KneighborsRegressorModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "KNeighborsRegressor"  # Define o nome para carregar parâmetros

        # Se não houver parâmetros passados, carrega os parâmetros do JSON
        if not kwargs:
            kwargs = FileManager.carregar_parametros_do_json(self.name)  # Usando o método da classe FileManager

        # Cria o modelo com os parâmetros (se fornecidos ou do JSON)
        self.modelo = KNeighborsRegressor(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)