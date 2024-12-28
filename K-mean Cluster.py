import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# تحميل البيانات من ملف CSV
data = pd.read_csv('dataset.csv')

# استخراج الميزات (Feature1 و Feature2)
X = data[['feature1', 'feature2']].values

# تحديد عدد  (clusters)
k = 2

# تهيئة عشوائية لمراكز المجموعات (centroids)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

def distance(point1, point2):
    """حساب المسافة  بين نقطتين"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def assign_clusters(X, centroids):
    """تحديد إلى أي مجموعة تنتمي كل نقطة"""
    clusters = []
    for point in X:
        distances = [distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(X, clusters, k):
    """تحديث مواقع مراكز المجموعات"""
    new_centroids = []
    for i in range(k):
        points_in_cluster = X[clusters == i]
        new_centroid = points_in_cluster.mean(axis=0) if len(points_in_cluster) > 0 else np.random.rand(X.shape[1])
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# تطبيق الخوارزمية
max_iterations = 100
for _ in range(max_iterations):
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, clusters, k)
    # إنهاء الخوارزمية إذا لم تعد مراكز المجموعات تتغير
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# عرض النتائج
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', label='Clusters')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering (Using CSV Data)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

