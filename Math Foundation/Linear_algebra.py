import numpy as np

"""
线性代数-数据的基本骨架
1.矩阵运算: 数据的表格
矩阵像Excel表格,用来组织数字
"""
# 创建2x2矩阵
matrix = np.array([[1, 2],
                   [3, 4]])

# 矩阵加法
matrix + 2  # 所有元素加2 → [[3,4],[5,6]]

# 矩阵乘法
np.dot(matrix, matrix)  # 矩阵自乘 → [[7,10],[15,22]]

# print(matrix + 2)
# print(np.dot(matrix, matrix))

"""
2.向量空间:箭头指向的方向
"""
# 在三维空间中的两个向量
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# 向量的点积（投影）
dot_product = np.dot(vector_a, vector_b)  # 1×4 + 2×5 + 3×6 = 32

# 向量长度
length_a = np.linalg.norm(vector_a)  # √(1²+2²+3²) ≈ 3.74

# print(dot_product)
# print(length_a)

"""
3.特征值/特征向量: 矩阵的本质
当矩阵作用在特定向量上时不改变方向
"""

# 求矩阵的特征值和特征向量
matrix = np.array([[2, 1],
                   [1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("特征值:", eigenvalues)  # [3., 1.]
print("特征向量:\n", eigenvectors)  # [[ 0.707, -0.707], [0.707, 0.707]]

"""
奇异值分解(SVD): 数据的本质拆分
将任意矩阵分解为三个特殊矩阵相乘:
"""

# 图像压缩示例（实际应用中）
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt

# 加载小图像
image = resize(data.astronaut(), (100, 100))
gray_image = np.mean(image, axis=2)

# 进行奇异值分解
U, s, VT = np.linalg.svd(gray_image, full_matrices=False)

# 仅保留前20个特征重建图像
k = 20
reconstructed = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]

# 显示压缩前后对比
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_image, cmap='gray')
ax1.set_title('原始图像')
ax2.imshow(reconstructed, cmap='gray')
ax2.set_title('压缩后图像 (SVD)')
plt.show()