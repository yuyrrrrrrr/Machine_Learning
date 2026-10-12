import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def Linear_regression():
    """

    :return:
    """
    # 数据部分, 特征
    x = [
        [160],
        [166],
        [172],
        [174],
        [180]
    ]
    # 标签
    y = [56.3, 60.6, 65.1, 68.5, 75]

    #数据预处理
    # 模型实例化
    estimator = LinearRegression()
    estimator.fit(x,y)

    print("estimator.coef_--->", estimator.coef_)
    print("estimator.intercept_--->", estimator.intercept_)
    # 模型预测
    result = estimator.predict([[176]])
    print("result--->", result)


if __name__ == '__main__':
    Linear_regression()
    print(f'')