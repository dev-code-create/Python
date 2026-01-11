from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([
   #Income and Spending
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2], 
    [10, 4], 
    [10, 0]
])

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=1)
x_pca = pca.fit_transform(X_scaled)

print(x_pca)