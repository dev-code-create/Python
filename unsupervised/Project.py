import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Create Synthetic Data (Simulating Mall Customer Data)
# In a real project, use: df = pd.read_csv('Mall_Customers.csv')
np.random.seed(42)
data = {
    'Annual Income (k$)': np.random.randint(15, 130, 200),
    'Spending Score (1-100)': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)

# 2. Preprocessing: Feature Selection & Scaling
# We select the columns for Income and Spending Score
X = df.iloc[:, [0, 1]].values

# Scaling is crucial for distance-based algorithms like K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. The Elbow Method to find the optimal number of clusters (k)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Inertia)')
plt.show()

# 4. Training the K-Means model on the dataset
# Based on the elbow, we choose k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 5. Visualizing the Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_scaled[y_kmeans == 0, 0], y=X_scaled[y_kmeans == 0, 1], s=100, label='Cluster 1')
sns.scatterplot(x=X_scaled[y_kmeans == 1, 0], y=X_scaled[y_kmeans == 1, 1], s=100, label='Cluster 2')
sns.scatterplot(x=X_scaled[y_kmeans == 2, 0], y=X_scaled[y_kmeans == 2, 1], s=100, label='Cluster 3')
sns.scatterplot(x=X_scaled[y_kmeans == 3, 0], y=X_scaled[y_kmeans == 3, 1], s=100, label='Cluster 4')
sns.scatterplot(x=X_scaled[y_kmeans == 4, 0], y=X_scaled[y_kmeans == 4, 1], s=100, label='Cluster 5')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids', marker='X', edgecolors='black')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.show()