import numpy as np

from math import sqrt
from collections import Counter

from .metrics import accuracy_score


class KNNClassifier:
    def __init__(self, k):
        '''初始化分类器'''
        assert k >= 1, 'k must be valid'
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_trian, y_train):
        '''根据训练数据集X_train和y_train训练kNN分类器'''
        self._X_train = X_trian
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        '''给定预测数据集，返回表示X_predict的结果向量'''
        assert self._X_train is not None and self._y_train is not None, 'must be fit before predict'

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        '''预测单个数据'''
        distance = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distance)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        '''根据测试数据集X_test和y_test确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return f'KNN(k={self.k})'
