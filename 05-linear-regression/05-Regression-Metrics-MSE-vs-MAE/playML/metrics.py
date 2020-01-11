import numpy as np

from math import sqrt


def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    '''计算MSE'''
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    '''计算RMSE'''
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    '''计算MAE'''
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)
