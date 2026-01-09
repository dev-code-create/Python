import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# sample data
X = np.array([
   #Income and Spending
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2], 
    [10, 4], 
    [10, 0]
])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

print(labels)
print(centers)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1], marker='X')
plt.show()

