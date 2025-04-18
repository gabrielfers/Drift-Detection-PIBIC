from sklearn.linear_model import Lasso
from classes.superclasse.ModeloBase import ModeloBase


class LassoRegressionModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.modelo = Lasso(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)