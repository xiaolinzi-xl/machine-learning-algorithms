import numpy as np


def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    return np.sum(y_true == y_predict) / len(y_true)
