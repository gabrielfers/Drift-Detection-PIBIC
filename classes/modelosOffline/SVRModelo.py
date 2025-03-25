from sklearn.svm import SVR
from classes.superclasse.ModeloBase import ModeloBase


class SVRModelo(ModeloBase):
    def __init__(self, kernel="rbf", C=10, epsilon=0.1, gamma="scale"):
        super().__init__()
        self.modelo = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)