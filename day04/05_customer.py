import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings(action='ignore')


# 聚类分析用户分群
def dm01():
    dataset = pd.read_csv('data/customers.csv')
    dataset.info()
    print('dataset-->\n', dataset)

    X = dataset.iloc[:, [3, 4]]
    print('X-->\n', X)
    mysse = []
    mysscore = []

    # 评估聚类个数
    for i in range(2, 11):
        mykeans = KMeans(n_clusters=i)
        mykeans.fit(X)
        mysse.append(mykeans.inertia_)  # inertia 簇内误差平方和
        ret = mykeans.predict(X)
        mysscore.append(silhouette_score(X, ret))  # sc系数 聚类需要1个以上的类别

    plt.plot(range(2, 11), mysse)
    plt.title('the elbow method')
    plt.xlabel('number of clusters')
    plt.ylabel('mysse')
    plt.grid()
    plt.show()

    plt.title('sh')
    plt.plot(range(2, 11), mysscore)
    plt.grid(True)
    plt.show()
    pass


def dm02():
    dataset = pd.read_csv('data/customers.csv')
    X = dataset.iloc[:, [3, 4]]
    mykeans = KMeans(n_clusters=5)
    mykeans.fit(X)
    y_kmeans = mykeans.predict(X)

    # 把类别是0的, 第0类数据,第1列数据, 作为x/y, 传给plt.scatter函数
    plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s=100, c='red', label='Standard')
    # 把类别是1的, 第0类数据,第1列数据, 作为x/y, 传给plt.scatter函数
    plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s=100, c='blue', label='Traditional')
    # 把类别是2的, 第0类数据,第1列数据, 作为x/y, 传给plt.scatter函数
    plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s=100, c='green', label='Normal')
    plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1], s=100, c='cyan', label='Youth')
    plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1], s=100, c='magenta', label='TA')
    plt.scatter(mykeans.cluster_centers_[:, 0], mykeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')

    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dm01()
    dm02()