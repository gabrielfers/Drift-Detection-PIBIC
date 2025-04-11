from sklearn.neighbors import KNeighborsRegressor
from classes.superclasse.ModeloBase import ModeloBase


class KNeighborsRegressorModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.modelo = KNeighborsRegressor(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)