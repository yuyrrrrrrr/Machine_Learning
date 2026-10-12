def dm00():
    """
    读取数据到本地（需科学上网）
    :return:
    """
    import pandas as pd

    # 1. 定义数据的网络地址和本地保存路径
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    local_file_path = "../data/boston_housing_local.csv"

    # 2. 从网络读取数据
    print(f"正在从 {url} 读取数据...")
    try:
        df = pd.read_csv(url)
        print("数据读取成功！")
    except Exception as e:
        print(f"读取数据时出错: {e}")
        # 如果网络请求失败，可以在这里添加备用方案
        exit()

    # 3. 显示数据的基本信息，确认读取正确
    print("\n数据基本信息：")
    print(f"形状: {df.shape}")  # 应该是 (506, 14)
    print(f"列名: {df.columns.tolist()}")
    print("\n前5行数据预览：")
    print(df.head())

    # 4. 将DataFrame保存到本地CSV文件
    df.to_csv(local_file_path, index=False)
    print(f"\n数据已成功保存到本地文件: {local_file_path}")


def dm01():
    import pandas as pd

    # 1. 定义数据的网络地址和本地保存路径
    # UCI 数据源地址
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    local_file_path = "../data/boston_housing_local.csv"

    # 2. 从网络读取数据
    # 数据集的 14 个列名（13 个特征 + 1 个目标变量 medv）
    column_names = [
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad",
        "tax", "ptratio", "b", "lstat", "medv"
    ]
    # 读取数据：该文件以空白字符分隔，且没有表头
    df = pd.read_csv(url, delim_whitespace=True, names=column_names)

    # 3. 显示数据的基本信息，确认读取正确
    print("\n数据基本信息：")
    print(f"形状: {df.shape}")  # 应该是 (506, 14)
    print(f"列名: {df.columns.tolist()}")
    print("\n前5行数据预览：")
    print(df.head())

    # 4. 将DataFrame保存到本地CSV文件
    df.to_csv(local_file_path, index=False)
    print(f"\n数据已成功保存到本地文件: {local_file_path}")


def boston_price_predict():
    """
    正规方程实现波士顿房价预测案例
    :return:
    """

    import pandas as pd
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 1. 准备数据
    local_file = "../data/boston_housing_local.csv"
    df = pd.read_csv(local_file)

    data = df.iloc[:, :-1].to_numpy()  # 除最后一列外全部作为特征
    target = df.iloc[:, -1].to_numpy()  # 最后一列作为目标

    # 2. 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(  # 源码查看使用
        data,
        target,
        train_size=0.8,
        random_state=5
    )

    # 3. 特征工程

    transform = StandardScaler()
    # 基于transform对象，对训练集（fit_transform）和测试集（transform）分别进行转换
    X_train = transform.fit_transform(X=X_train)
    X_test = transform.transform(X=X_test)

    # 4. 模型训练
    estimator = LinearRegression()
    estimator.fit(X_train, y_train)

    # 5. 模型预测
    y_pred = estimator.predict(X_test)
    print("y_pred--->", y_pred)
    print("y_test--->", y_test)

    # 6. 模型评估
    loss = mean_squared_error(y_test, y_pred)
    print("loss--->", loss)


if __name__ == '__main__':
    # 1. 数据加载
    # 任选一个运行就可以
    #dm00()
    dm01()

    # 2. 数据加载成功后再运行
    boston_price_predict()