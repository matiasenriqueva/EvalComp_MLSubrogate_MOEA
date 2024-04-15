from sklearn.linear_model import SGDRegressor
from skmultiflow.meta import RegressorChain

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RegressorChainSurrogate():
    def __init__(self):
        self.rc = RegressorChain(SGDRegressor(loss='squared_error', random_state=1))
        self.scaler = StandardScaler()

    def evaluate(self, X):
        X = self.scaler.transform(X)
        return self.rc.predict(X)

    def fit(self, data):
        data = self.scaler.fit_transform(data)
        
        self.rc.fit(X, y)