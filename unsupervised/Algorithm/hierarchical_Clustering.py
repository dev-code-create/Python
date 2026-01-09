from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt

X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

model = AgglomerativeClustering(n_clusters=2)

labels = model.fit_predict(X)

linked = linkage(X , method='ward')\

dendrogram(linked)
plt.show()


print(labels)