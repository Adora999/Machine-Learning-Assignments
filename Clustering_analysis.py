import os  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.environ['OMP_NUM_THREADS'] = '1'  # Corrected typo in 'os.environ'

data = pd.DataFrame({
    'x': [5, 4, 5, 4, 5, 4, 10, 11, 11, 10, 12, 10, 20, 22, 21, 20, 22, 21],
    'y': [40, 41, 42, 43, 45, 41, 60, 61, 62, 63, 64, 65, 71, 72, 73, 74, 75, 76],
    'z': [100, 101, 102, 103, 100, 102, 200, 201, 203, 208, 210, 205, 305, 300, 301, 302, 304, 303]
})

scaler = StandardScaler()  # Fixed typo in variable name
data_scaled = scaler.fit_transform(data)  # Corrected comment typo

distance_matrix = linkage(data_scaled, method='ward')  # Fixed spacing
dendrogram(distance_matrix)  # Generates the dendrogram
plt.title("Dendrogram for Hierarchical Clustering")  # Title of the dendrogram
plt.xlabel("Data Points")  # X-axis label
plt.ylabel("Euclidean Distance")  # Y-axis label
plt.show()

inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Added missing parameters
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)  # Computes the sum of squared distances

plt.plot(k_values, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Groups the data into 3 clusters and adds these cluster labels as a new column in the dataset
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Fixed typo in function call
data['KMeans_Cluster'] = kmeans.fit_predict(data_scaled)

h_clusters = fcluster(distance_matrix, optimal_k, criterion='maxclust')  # Fixed typo in variable name
data['Hierarchical_Cluster'] = h_clusters

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Fixed typo in figsize
sns.scatterplot(x=data['x'], y=data['y'], hue=data['KMeans_Cluster'], palette='viridis', ax=axes[0])
axes[0].set_title('KMeans Clustering')
sns.scatterplot(x=data['x'], y=data['y'], hue=data['Hierarchical_Cluster'], palette='Set1', ax=axes[1])
axes[1].set_title('Hierarchical Clustering')
plt.show()