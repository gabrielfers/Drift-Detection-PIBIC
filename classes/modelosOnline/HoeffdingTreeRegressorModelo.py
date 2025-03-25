from sklearn import tree
from classes.superclasse.ModeloBase import ModeloBase

class HoeffdingTreeRegressorModelo(ModeloBase):
    def __init__(self, grace_period=50, leaf_prediction="adaptive"):
        super().__init__()
        self.modelo = tree.HoeffdingTreeRegressor(grace_period=grace_period, leaf_prediction=leaf_prediction)

    def treinar(self, X, y):
        for i in range(len(X)):
            X_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}
            self.modelo.learn_one(X_dict, y[i][0])

    def prever(self, X):
        X_dict = {f"t{i+1}": value for i, value in enumerate(X[0])}
        return self.modelo.predict_one(X_dict)  # Retorna o valor escalar diretamente