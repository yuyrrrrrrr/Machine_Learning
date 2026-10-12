import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, binom, poisson

"""
概率统计-预测与不确定性的艺术
1.概率分布:事件发生的可能性
"""

# 正态分布（高斯分布）
x = np.linspace(-5, 5, 100)
plt.plot(x, norm.pdf(x, 0, 1), label='正态分布')

# 二项分布（抛硬币）
n, p = 10, 0.5
x_binom = np.arange(0, 11)
plt.stem(x_binom, binom.pmf(x_binom, n, p), 'bo', label='二项分布')

# 泊松分布（罕见事件）
lambda_ = 3
x_poisson = np.arange(0, 10)
plt.stem(x_poisson, poisson.pmf(x_poisson, lambda_), 'g^', label='泊松分布')

plt.legend()
plt.title('常见概率分布')
plt.xlabel('数值')
plt.ylabel('概率密度')
plt.show()

"""
2.贝叶斯定理: 新证据更新信念
医生诊断疾病的情景：假设：

-疾病D患病率: 1% → P(D) = 0.01
-检测灵敏度: 99% → P(阳性|D) = 0.99
-检测特异度: 95% → P(阴性|健康) = 0.95
求P(确实有病|检测阳性)?
"""

# 计算贝叶斯概率
p_disease = 0.01      # P(D)
p_positive_given_disease = 0.99  # P(阳性|D)
p_negative_given_healthy = 0.95  # P(阴性|健康)

# P(阳性|健康) = 1 - P(阴性|健康)
p_positive_given_healthy = 1 - p_negative_given_healthy

# P(阳性) = P(阳性|D) * P(D) + P(阳性|健康) * P(健康)
p_positive = (p_positive_given_disease * p_disease) + (p_positive_given_healthy * (1-p_disease))

# P(D|阳性) = [P(阳性|D) * P(D)] / P(阳性)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"检测阳性后真正患病的概率: {p_disease_given_positive*100:.2f}%")  # ≈16.2%

"""
3.假设检验: 判断差异是否真实
学生A和B谁成绩更好
-A班平均分：78分（30人）
-B班平均分：82分（30人） 差异显著吗？
"""

from scipy import stats

# 生成模拟数据（方差为10）
np.random.seed(42)
class_a = np.random.normal(78, 10, 30)
class_b = np.random.normal(82, 10, 30)

# 进行t检验
t_stat, p_value = stats.ttest_ind(class_a, class_b)

alpha = 0.05  # 显著性水平
if p_value < alpha:
    print(f"p值 = {p_value:.4f} < 0.05，两组有显著差异")
else:
    print(f"p值 = {p_value:.4f} >= 0.05，两组无显著差异")

"""
4.回归分析: 预测趋势
根据房屋面积预测价格:
"""

from sklearn.linear_model import LinearRegression

# 样本数据（面积 vs 价格）
areas = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)  # m²
prices = np.array([200, 240, 290, 340, 380])  # 万元

# 创建模型并拟合
model = LinearRegression()
model.fit(areas, prices)

# 预测80平米房子的价格
prediction = model.predict([[80]])
print(f"预测80平米房屋价格：{prediction[0]:.1f}万元")

# 绘制数据点及拟合线
plt.scatter(areas, prices, label='实际价格')
plt.plot(areas, model.predict(areas), 'r-', label='预测趋势')
plt.scatter([80], prediction, c='g', marker='*', s=200, label='预测点')
plt.xlabel('面积(m²)')
plt.ylabel('价格(万元)')
plt.legend()
plt.show()