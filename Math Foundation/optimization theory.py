"""
优化理论-寻找最佳解决方案
1.梯度下降: 一步一步找到最低点
寻找函数最低点:
"""
import numpy as np
from matplotlib import pyplot as plt


# 定义函数：f(x) = x^4 - 3x^3 + 2
def f(x):
    return x ** 4 - 3 * x ** 3 + 2


# 导数：f'(x) = 4x^3 - 9x^2
def df(x):
    return 4 * x ** 3 - 9 * x ** 2


# 梯度下降
x = 2.0  # 初始点
lr = 0.01  # 学习率
steps = 50  # 迭代次数

# 记录路径
path = [x]

for i in range(steps):
    grad = df(x)
    x = x - lr * grad  # 向下走一步
    path.append(x)

# 绘制函数及下降路径
x_vals = np.linspace(-1, 3, 200)
plt.plot(x_vals, f(x_vals), 'b-', lw=2, label='f(x)')
plt.scatter(path, f(np.array(path)), c='r', marker='o')
for i in range(1, len(path)):
    plt.annotate('', xy=(path[i], f(path[i])),
                 xytext=(path[i - 1], f(path[i - 1])),
                 arrowprops=dict(arrowstyle='->', color='r'))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('梯度下降过程')
plt.grid(True)
plt.show()


"""
约束优化: 带限制的最优化问题
"""
from scipy.optimize import minimize

# 目标函数：f(x,y) = (x-1)² + (y-2.5)²
objective = lambda x: (x[0]-1)**2 + (x[1]-2.5)**2

# 约束条件：
# x - 2y >= -1    → 约束1
# -x - 2y >= -6   → 约束2
# -x + 2y >= -2   → 约束3
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 2*x[1] + 1},  # ≥0
    {'type': 'ineq', 'fun': lambda x: -x[0] - 2*x[1] + 6},
    {'type': 'ineq', 'fun': lambda x: -x[0] + 2*x[1] + 2}
]

# 初始猜测
x0 = [0, 0]

# 求解
solution = minimize(objective, x0, constraints=constraints)
print(f"最小值点: ({solution.x[0]:.2f}, {solution.x[1]:.2f})")
print(f"最小值: {solution.fun:.4f}")


"""
凸优化基础: 不会陷入局部最优的特例
"""
"""
graph LR
    A[优化问题] --> B{是否为凸？}
    B -- 是 --> C[只有一个全局最优解]
    B -- 否 --> D[可能有多个局部最优解]
    
    subgraph 凸函数特性
    C --> E[二阶导数>=0]
    C --> F[任意连线位于函数上方]
    C --> G[局部最优即全局最优]
    end
    
    
    凸优化的黄金定律：
        1.凸问题总能找到全局最优解
        2.机器学习中常将非凸问题转化为凸问题求解
"""

"""
4.学习率策略: 智能调整学习步伐
不同学习率策略对比:
"""
# 三种学习率策略
def constant_lr(epoch):  # 固定学习率
    return 0.1

def step_lr(epoch):     # 阶梯下降
    if epoch < 10:
        return 0.1
    elif epoch < 20:
        return 0.01
    else:
        return 0.001

def exp_lr(epoch):      # 指数衰减
    return 0.1 * (0.9 ** epoch)

# 绘制学习率变化曲线
epochs = range(1, 31)

plt.plot(epochs, [constant_lr(e) for e in epochs], 'b-o', label='固定学习率')
plt.plot(epochs, [step_lr(e) for e in epochs], 'r-s', label='阶梯衰减')
plt.plot(epochs, [exp_lr(e) for e in epochs], 'g-^', label='指数衰减')
plt.xlabel('训练轮次(epoch)')
plt.ylabel('学习率')
plt.title('不同学习率策略比较')
plt.legend()
plt.grid(True)
plt.show()


"""
数学在AI中的实际应用
典型AI任务中涉及的数学：

AI模型	线性代数	概率统计	微积分	优化方法
线性回归	★★	    ★★	    ★	    ★★
神经网络	★★★	★	    ★★★	★★★
推荐系统	★★	    ★★★	★	    ★★
图像处理	★★★	★	    ★	    ★★
强化学习	★	    ★★★	★★	    ★★★
学习建议：
1. 理解 > 记忆：先搞懂概念，公式自然记住
2. 可视化是利器：多画图帮助理解抽象概念
3. 动手计算：Python工具包是数学学习好帮手
4. 实际应用驱动：关注知识在AI中的具体用途
"""