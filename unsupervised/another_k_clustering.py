import numpy as np
import matplotlib.pyplot as plt

# Dataset (Math, Science scores)
X = np.array([
    [35, 30],
    [40, 35],
    [45, 40],
    [80, 85],
    [85, 80],
    [90, 88],
    [60, 55],
    [65, 60],
    [70, 65]
])

k = 3  # number of clusters
iterations = 10

# Step 1: Initialize centroids randomly
np.random.seed(42)
centroids = X[np.random.choice(len(X), k, replace=False)]

for _ in range(iterations):
    clusters = []

    # Step 2: Assign points to nearest centroid
    for point in X:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters.append(cluster_index)

    clusters = np.array(clusters)

    # Step 3: Update centroids
    new_centroids = []
    for i in range(k):
        new_centroids.append(X[clusters == i].mean(axis=0))

    centroids = np.array(new_centroids)

# Plot result
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)
plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.title("K-Means Clustering (From Scratch)")
plt.show()
