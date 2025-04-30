from sklearn.linear_model import LinearRegression
from regressores.ModeloBase import ModeloBase


class LinearRegressionModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.modelo = LinearRegression(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        return self.modelo.predict(X)