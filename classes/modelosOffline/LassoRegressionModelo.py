from sklearn.linear_model import Lasso
from classes.superclasse.ModeloBase import ModeloBase


class LassoRegressionModelo(ModeloBase):
    def __init__(self, alpha=0.01, max_iter=1000):
        super().__init__()
        self.modelo = Lasso(alpha=alpha, max_iter=max_iter)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)