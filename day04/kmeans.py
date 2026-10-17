import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# # ========== 1. 生成数据 ==========
# # random_state:设置随机种子点
# x, y = make_blobs(
#     n_samples=1000,
#     n_features=2,
#     centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
#     cluster_std=[0.4, 0.2, 0.2, 0.2],
#     random_state=10
# )
#
# print("x --- >", x[:3])
# print("y --- >", y[:3])
#
# # ========== 2. 训练 KMeans 模型 ==========
# # 我们知道真实类别是 4 类，所以 n_clusters=4
# kmeans = KMeans(
#     n_clusters=4,
#     init='k-means++',   # 优化初始质心选择，避免陷入局部最优
#     n_init=10,          # 多次运行取最优
#     max_iter=300,
#     random_state=10
# )
# y_pred = kmeans.fit_predict(x)
#
# # 获取聚类中心和标签
# centers = kmeans.cluster_centers_
# labels = kmeans.labels_
# inertia = kmeans.inertia_     # 簇内平方和（越小越好）
# print("聚类中心:\n", centers)
# print("簇内平方和 (inertia):", inertia)
#
# # ========== 3. 可视化对比 ==========
# plt.figure(figsize=(12, 5))
#
# # 3.1 真实标签分布
# plt.subplot(1, 2, 1)
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', s=20, alpha=0.7)
# plt.title("真实类别 (Ground Truth)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
#
# # 3.2 KMeans 聚类结果
# plt.subplot(1, 2, 2)
# plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.7)
# # 画出聚类中心
# plt.scatter(centers[:, 0], centers[:, 1],
#             c='red', marker='X', s=200,
#             edgecolors='black', label='Centroids')
# plt.title(f"KMeans 聚类结果 (inertia={inertia:.2f})")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
#
# plt.tight_layout()
# plt.show()

def dm01_kmeans():

    # 1. 准备散据
    #n_sumples: 种本数量
    # n_features:特征数量
    # centers:每臣数据前中心点
    # cluster_std:每组数据的标准差
    # random_state : RNNMATA
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=10
    )

    print("x --- >", x[:3])
    print("y --- >", y[:3])

    # naker: *--- >五角晕:H --- >六边形
    #x[:,0]:数据的第一列特征
    # x[;,1]:数据的第二列特征
    plt.scatter(x[:,0], x[:, 1], marker="*")
    plt.show()

if __name__ == '__main__':
    dm01_kmeans()
    print("ond")