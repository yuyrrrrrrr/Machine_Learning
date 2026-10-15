# 导入工具包
from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率


# 1. 定义函数 dm01_loadiris(), 加载数据集.
def dm01_loadiris():
    # 1. 加载数据集, 查看数据
    iris_data = load_iris()
    print(type(iris_data))           # 字典形式, 键: 属性名, 值: 数据.
    print(iris_data.keys())

    # 1.1 查看数据集
    print(iris_data.data[:5])
    # 1.2 查看目标值.
    print(iris_data.target)
    # 1.3 查看目标值名字.
    print(iris_data.target_names)
    # 1.4 查看特征名.
    print(iris_data.feature_names)
    # 1.5 查看数据集的描述信息.
    print(iris_data.DESCR)
    # 1.6 查看数据文件路径
    print(iris_data.filename)

# 2. 定义函数 dm02_showiris(), 显示鸢尾花数据.
def dm02_showiris():
    # 1. 加载数据集, 查看数据
    iris_data = load_iris()
    # 2. 数据展示
    # 读取数据, 并设置 特征名为列名.
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    # print(iris_df.head(5))
    iris_df['label'] = iris_data.target

    # 可视化, x=花瓣长度, y=花瓣宽度, data=iris的df对象, hue=颜色区分, fit_reg=False 不绘制拟合回归线.
    sns.lmplot(x='petal length (cm)', y='petal width (cm)', data=iris_df, hue='label', fit_reg=False)
    plt.title('iris data')
    plt.show()


# 3. 定义函数 dm03_train_test_split(), 实现: 数据集划分
def dm03_train_test_split():
    # 1. 加载数据集, 查看数据
    iris_data = load_iris()
    # 2. 划分数据集, 即: 特征工程(预处理-标准化)
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=22)
    print(f'数据总数量: {len(iris_data.data)}')
    print(f'训练集中的x-特征值: {len(x_train)}')
    print(f'训练集中的y-目标值: {len(y_train)}')
    print(f'测试集中的x-特征值: {len(x_test)}')

# 4. 定义函数 dm04_模型训练和预测(), 实现: 模型训练和预测
def dm04_model_train_and_predict():
    # 1. 加载数据集, 查看数据
    iris_data = load_iris()

    # 2. 划分数据集, 即: 数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)

    # 3. 数据集预处理-数据标准化(即: 标准的正态分布的数据集)
    transfer = StandardScaler()
    # fit_transform(): 适用于首次对数据进行标准化处理的情况，通常用于训练集, 能同时完成 fit() 和 transform()。
    x_train = transfer.fit_transform(x_train)
    # transform(): 适用于对测试集进行标准化处理的情况，通常用于测试集或新的数据. 不需要重新计算统计量。
    x_test = transfer.transform(x_test)

    # 4. 机器学习(模型训练)
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train, y_train)

    # 5. 模型评估.
    # 场景1: 对抽取出的测试集做预测.
    # 5.1 模型评估, 对抽取出的测试集做预测.
    y_predict = estimator.predict(x_test)
    print(f'预测结果为: {y_predict}')

    # 场景2: 对新的数据进行预测.
    # 5.2 模型预测, 对测试集进行预测.
    # 5.2.1 定义测试数据集.
    my_data = [[5.1, 3.5, 1.4, 0.2]]
    # 5.2.2 对测试数据进行-数据标准化.
    my_data = transfer.transform(my_data)
    # 5.2.3 模型预测.
    my_predict = estimator.predict(my_data)
    print(f'预测结果为: {my_predict}')

    # 5.2.4 模型预测概率, 返回每个类别的预测概率
    my_predict_proba = estimator.predict_proba(my_data)
    print(f'预测概率为: {my_predict_proba}')

    # 6. 模型预估, 有两种方式, 均可.
    # 6.1 模型预估, 方式1: 直接计算准确率, 100个样本中模型预测正确的个数.
    my_score = estimator.score(x_test, y_test)
    print(my_score)  # 0.9666666666666667

    # 6.2 模型预估, 方式2: 采用预测值和真实值进行对比, 得到准确率.
    print(accuracy_score(y_test, y_predict))


# 在main方法中测试.
if __name__ == '__main__':
    # 1. 调用函数 dm01_loadiris(), 加载数据集.
    dm01_loadiris()

    # 2. 调用函数 dm02_showiris(), 显示鸢尾花数据.
    # dm02_showiris()

    # 3. 调用函数 dm03_train_test_split(), 查看: 数据集划分
    # dm03_train_test_split()

    # 4. 调用函数 dm04_模型训练和预测(), 实现: 模型训练和预测
    # dm04_model_train_and_predict()