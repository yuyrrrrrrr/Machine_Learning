# 0.导包
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

# 1.加载数据
from pathlib import Path
# boston = load_boston()
# print(boston)
import pandas as pd
import numpy as np


def dm01():
    BASE_DIR = Path(__file__).resolve().parent
    local_file = BASE_DIR.parent / "data" / "boston_housing_local.csv"
    df = pd.read_csv(local_file)

    data = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    data = data.to_numpy()
    # print(data)
    # print(type(data))

    target = target.to_numpy()
    # print(target)
    # print(type(target))
    #
    # print(data.max())
    # print(data.min())

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=5
    )
    # print(X_train)
    # print(X_train.shape)
    # print(X_test.shape)

    # 特征工程
    # 实例化, 标准化
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.fit_transform(X_test)
    # print(X_train.max())
    # print(X_train.min())
    # print(X_test.max())
    # print(X_test.min())

    estimator = LinearRegression()
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    # print(estimator.coef_)
    # print(estimator.intercept_)
    # print(y_pred)


    # 实例化梯度下降模型
    estimator = SGDRegressor(max_iter=5, )

    loss = mean_squared_error(y_test, y_pred)
    print(loss)
if __name__ == '__main__':
    dm01()
    print("end")