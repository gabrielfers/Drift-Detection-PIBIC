from sklearn.neural_network import MLPRegressor
from classes.superclasse.ModeloBase import ModeloBase

class MLPRegressorModelo(ModeloBase):
    def __init__(self, hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                 learning_rate_init=0.001, max_iter=500, random_state=42):
        super().__init__()
        self.modelo = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                     learning_rate_init=learning_rate_init, max_iter=max_iter, random_state=random_state)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)