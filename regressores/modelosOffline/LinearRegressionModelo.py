from sklearn.linear_model import LinearRegression
from regressores.ModeloBase import ModeloBase
from utils.FileManager import FileManager


class LinearRegressionModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.name = "LinearRegression"  # Define antes de carregar os parâmetros

        # Se não houver parâmetros passados, carrega os parâmetros do JSON
        if not kwargs:
            kwargs = FileManager.carregar_parametros_do_json(self.name)  # Usando o método da classe FileManager

        self.modelo = LinearRegression(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        return self.modelo.predict(X)