from sklearn.linear_model import Ridge
from classes.superclasse.ModeloBase import ModeloBase

class RidgeRegressionModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.modelo = Ridge(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)