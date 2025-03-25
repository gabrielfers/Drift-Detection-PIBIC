from sklearn.neighbors import KNeighborsRegressor
from classes.superclasse.ModeloBase import ModeloBase


class KNeighborsRegressorModelo(ModeloBase):
    def __init__(self, n_neighbors=10, weights="distance", metric="minkowski"):
        super().__init__()
        self.modelo = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)