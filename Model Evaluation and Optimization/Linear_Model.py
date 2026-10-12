"""
监督学习:有老师的指导学习
1.线性回归: 预测连续值
-核心思想:找到一条最佳拟合线, 预测连续值结果
-实例应用: 根据房子面积预测房价
"""
# 简单线性回归示例
import numpy as np
from sklearn.linear_model import LinearRegression

# 房子面积数据（平方米）
house_sizes = np.array([50, 70, 90, 110]).reshape(-1, 1)
# 对应房价（万元）
prices = np.array([300, 400, 500, 600])

# 创建模型并训练
model = LinearRegression()
model.fit(house_sizes, prices)

# 预测120平房子的价格
prediction = model.predict([[120]])
print(f"预测房价: {prediction[0]:.1f}万元")  # 输出: 预测房价: 700.0万元

"""
2.逻辑回归: 解决二分类问题
-核心思想: 计算某件事发生的概率 (0-1之间)
-实例应用: 判断邮件是否为垃圾邮件
"""
# 垃圾邮件识别示例
from sklearn.linear_model import LogisticRegression

# 假设有以下特征：
# feature1: 邮件包含"免费"次数
# feature2: 邮件包含"获奖"次数
X_train = [[3, 1], [5, 2], [1, 0], [0, 1]]  # 训练数据
y_train = [1, 1, 0, 0]  # 1=垃圾邮件，0=正常邮件

# 创建模型并训练
spam_detector = LogisticRegression()
spam_detector.fit(X_train, y_train)

# 预测新邮件
new_email = [[4, 3]]  # 包含4次"免费"，3次"获奖"
prediction = spam_detector.predict(new_email)
print("垃圾邮件" if prediction[0] == 1 else "正常邮件")  # 输出: 垃圾邮件