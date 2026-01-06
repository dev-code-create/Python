import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset (2D points)
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 5]
])

# Create KMeans model
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("K-Means Clustering Example")
plt.show()
