import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        '''根据训练数据集X获取数据的均值和标准差'''
        assert X.ndim == 2, ''
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        assert X.ndim == 2, ''
        assert self.mean_ is not None and self.scale_ is not None, ''

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX

    def __repr__(self):
        return f'mean_ = {self.mean_}, scale_ = {self.scale_}'
