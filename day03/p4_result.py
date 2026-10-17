from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
from sklearn.model_selection import train_test_split, GridSearchCV    # 分割训练集和测试集的, 网格搜索的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率


# 1. 获取数据集.
iris_data = load_iris()

# 2. 数据基本处理-划分数据集.
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)

# 3. 数据集预处理-数据标准化.
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练.
# 4.1 创建估计器对象.
estimator = KNeighborsClassifier()
# 4.2 使用校验验证网格搜索.  指定参数范围.
param_grid = {"n_neighbors": range(1, 10)}
# 4.3 具体的 网格搜索过程 + 交叉验证.
# 参1: 估计器对象, 参2: 参数范围, 参3: 交叉验证的折数.
estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
# 具体的模型训练过程.
estimator.fit(x_train, y_train)


# 4.4 交叉验证, 网格搜索结果查看.
print(estimator.best_score_)       # 模型在交叉验证中, 所有参数组合中的最高平均测试得分
print(estimator.best_estimator_)   # 最优的估计器对象.
print(estimator.cv_results_)       # 模型在交叉验证中的结果.
print(estimator.best_params_)      # 模型在交叉验证中的结果.


# 5. 得到最优模型后, 对模型重新预测.
estimator = KNeighborsClassifier(n_neighbors=6)
estimator.fit(x_train, y_train)
print(f'模型评估: {estimator.score(x_test, y_test)}')   # 因为数据量和特征的问题, 该值可能小于上述的平均测试得分.