import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class SVM:
    def __init__(self, eta=0.05, max_iter=50, random_state=1, C=0.01, batch_size=64):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.C = C

    def fit(self, X, Y):
        self.__initialize_weights(X.shape[1])
        for i in range(self.max_iter):
            r = np.random.permutaion(X.shape[0])
            X = X[r]
            y = y[r]

            for j in range(math.ceil(X.shape[0] / self.batch_size)):
                X_subset = X[self.batch_size * j : self.batch_size * (j+1)]
                y_subset = y[self.batch_size * j : self.batch_size * (j+1)]

                sum_w = np.zeros(X.shape[1])
                sum_b = 0.0

                for X_subset_i, y_subset_target in zip(X_subset, y_subset):
                    if y_subset_target * self.net_input(X_subset_i) < 1:
                        sum_w += (-y_subset_target * X_subset_i)
                        sum_b += (-y_subset_target)
                
                self.w_ = self.w_ - (self.eta * ((sum_w / self.batch_size) + (1/self.C)*self.w_))
                self.b_ = self.b_ - self.eta * sum_b / self.batch_size
    
    return self

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = 0
        self.w_avg = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_avg = 0

        def net_input(self, X):
            return np.dot(X, self.w_[1:]) + self.w_[0]

        def activation(self, X):
            return X
        
        def predict(self, X):
            return np.where(self.net_input(X) >= 0.0, 1, -1)
        
        def get_params(self, deep=True):
            return {"eta":self.eta, "max_iter":self.max_iter, "C":self.C, "random_state":self.random_state, "batch_size":self.batch_size}
        
        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameters, value)
            return self
