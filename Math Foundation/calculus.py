"""
微积分-变化的数学语言
1.导数与积分: 变化与累积
导数 ≈ 瞬时速度, 积分 ≈ 总距离
"""
import numpy as np
from matplotlib import pyplot as plt

# 某车辆的运动函数：位置 = 时间²
t = np.linspace(0, 5, 100)  # 0到5秒
position = t**2              # 位置函数

# 计算导数（速度）
# 导数的数值计算：dy/dx ≈ Δy/Δx
velocity = np.gradient(position, t)  # 2t

# 计算积分（总路程）
# 积分的数值计算（累加）
distance = np.cumsum(velocity * np.diff(t, prepend=0))

# 绘制结果
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(t, position, 'b-', label='位置')
plt.plot(t, velocity, 'g--', label='速度(导数)')
plt.legend()
plt.title('位置与速度关系')

plt.subplot(212)
plt.plot(t, distance, 'r-', label='路程(积分)')
plt.legend()
plt.xlabel('时间(秒)')
plt.show()

"""
偏导数: 多维空间的变化率
温度场的变化 (随时间 + 位置):
"""
from mpl_toolkits.mplot3d import Axes3D

# 创建时间和空间的网格
x = np.linspace(0, 10, 100)  # 空间坐标
t = np.linspace(0, 5, 100)    # 时间坐标
X, T = np.meshgrid(x, t)

# 温度函数：温度 = e^{-0.1t} * sin(x)
Z = np.exp(-0.1*T) * np.sin(X)

# 绘制3D温度场
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, Z, cmap='viridis')
ax.set_xlabel('位置(x)')
ax.set_ylabel('时间(t)')
ax.set_zlabel('温度(℃)')
ax.set_title('空间温度分布随时间变化')
plt.show()

"""
梯度: 最陡的上山方向
"""
# 定义一个山峰形状的函数
def mountain(x, y):
    return np.exp(-0.1*(x**2 + y**2)) * np.cos(0.5*x)

# 创建网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = mountain(X, Y)

# 计算梯度（下山方向）
gy, gx = np.gradient(Z)
skip = 5  # 显示部分箭头

# 绘制等高线图
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, 20, cmap='viridis')
plt.colorbar()
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           -gx[::skip, ::skip], -gy[::skip, ::skip],
           scale=50, color='white')  # 负梯度表示最陡下降方向
plt.title('地形梯度图 - 白色箭头指向最陡下降方向')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

"""
4.泰勒级数: 用多项式逼近复杂函数
用多项式逼近正弦函数:
"""
# 正弦函数及其泰勒展开
x = np.linspace(-10, 10, 500)
sin_x = np.sin(x)

# 不同阶数的泰勒展开
taylor1 = x  # 1阶
taylor3 = x - x**3/6  # 3阶
taylor5 = taylor3 + x**5/120  # 5阶

# 绘制比较图
plt.figure(figsize=(10, 6))
plt.plot(x, sin_x, 'b-', lw=3, label='真实 sin(x)')
plt.plot(x, taylor1, 'g--', label='1阶展开')
plt.plot(x, taylor3, 'r-.', label='3阶展开')
plt.plot(x, taylor5, 'm:', lw=2, label='5阶展开')
plt.ylim(-3, 3)
plt.legend()
plt.title('泰勒级数逼近正弦函数')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()