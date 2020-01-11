import numpy as np


class SimpleLinearRegreession1:
    def __init__(self):
        '''初始化Simple Linear Regression 模型'''
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''根据训练数据集训练Simple Linear Regression模型'''
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        '''给定待预测数据集，返回表示x_predict的结果向量'''
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''给定单个数据，返回x_single的预测结果值'''
        return self.a_ * x_single + self.b_
