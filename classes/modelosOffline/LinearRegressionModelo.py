from sklearn.linear_model import LinearRegression
from classes.superclasse.ModeloBase import ModeloBase


class LinearRegressionModelo(ModeloBase):
    def __init__(self):
        super().__init__()
        self.modelo = LinearRegression()

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)