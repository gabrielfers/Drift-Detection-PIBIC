from sklearn.linear_model import Ridge
from classes.superclasse.ModeloBase import ModeloBase

class RidgeRegressionModelo(ModeloBase):
    def __init__(self, alpha=1.0, solver="auto"):
        super().__init__()
        self.modelo = Ridge(alpha=alpha, solver=solver)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)