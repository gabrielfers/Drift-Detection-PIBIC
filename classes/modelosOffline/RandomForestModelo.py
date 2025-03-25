from sklearn.ensemble import RandomForestRegressor
from classes.superclasse.ModeloBase import ModeloBase

class RandomForestModelo(ModeloBase):
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, random_state=42):
        super().__init__()
        self.modelo = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                             min_samples_split=min_samples_split, random_state=random_state)

    def treinar(self, X, y):
        self.modelo.fit(X, y)

    def prever(self, X):
        return self.modelo.predict(X)