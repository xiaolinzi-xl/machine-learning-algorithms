import numpy as np
from .metrics import r2_score


class SimpleLinearRegreession:
    def __init__(self):
        '''初始化Simple Linear Regression 模型'''
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''根据训练数据集训练Simple Linear Regression模型'''
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        '''给定待预测数据集，返回表示x_predict的结果向量'''
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''给定单个数据，返回x_single的预测结果值'''
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        '''根据测试数据集x_test和y_test确定当前模型的准确度'''
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return f'a:{self.a_}, b:{self.b_}'
