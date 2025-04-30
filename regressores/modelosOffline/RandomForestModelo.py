from sklearn.ensemble import RandomForestRegressor
from regressores.ModeloBase import ModeloBase

class RandomForestModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.modelo = RandomForestRegressor(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)