import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        X_norm = (X - self.min) / (self.max - self.min)
        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
