from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from classes.superclasse.ModeloBase import ModeloBase


class LinearRegressionModelo(ModeloBase, BaseEstimator, RegressorMixin):
    def __init__(self):
        super().__init__()
        self.__name__ = "LinearRegressionModelo"
        self.modelo = LinearRegression()

    def treinar(self, X, y):
        """Treina o modelo usando os dados fornecidos."""
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        """Faz previsões usando o modelo treinado."""
        return self.modelo.predict(X)