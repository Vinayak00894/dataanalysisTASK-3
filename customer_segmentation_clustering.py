# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load dataset (simulated data as actual file not available)
customer_data = pd.DataFrame({
    'Customer ID': range(1, 201),
    'Age': [25, 45, 29, 36, 22, 47, 52, 46, 31, 55] * 20,
    'Annual Income': [40000, 82000, 35000, 67000, 32000, 90000, 97000, 78000, 43000, 99000] * 20,
    'Spending Score': [39, 81, 6, 77, 40, 76, 94, 3, 72, 14] * 20
})

# Step 1b: Inspect dataset
print("Dataset Shape:", customer_data.shape)
print("Missing Values:\n", customer_data.isnull().sum())
print("Duplicates:", customer_data.duplicated().sum())
print("Data Types:\n", customer_data.dtypes)
print("\nSummary Statistics:\n", customer_data.describe())

# Step 2: Data Preprocessing - Standardize features for clustering
features = customer_data[['Age', 'Annual Income', 'Spending Score']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3a: Determine optimal clusters using Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

# Step 3b: Determine clustering quality using Silhouette Score
silhouette_scores = []
for k in range(2, 11):  # silhouette score not defined for k=1
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))

plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores For Various Cluster Counts')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Step 3c: Fit KMeans with chosen optimal clusters (k=4 here)
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans_final.fit_predict(scaled_features)

# Step 4a: Visualization - Reduce features to 2D with PCA for plotting
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
customer_data['PCA1'] = pca_components[:, 0]
customer_data['PCA2'] = pca_components[:, 1]

# Scatter plot of clusters in PCA space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segments Based on PCA of Features')
plt.show()

# Step 4b: Pairplot to visualize features colored by cluster
sns.pairplot(customer_data, vars=['Age', 'Annual Income', 'Spending Score'], hue='Cluster', palette='Set2')
plt.suptitle('Pairplot of Features by Cluster', y=1.02)
plt.show()

# Step 4c: Centroid visualization (original scale)
centroids_scaled = kmeans_final.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids, columns=['Age', 'Annual Income', 'Spending Score'])
print("\nCluster Centroids on Original Scale:\n", centroid_df)

# Final cluster summary by mean values
cluster_summary = customer_data.groupby('Cluster')[['Age', 'Annual Income', 'Spending Score']].mean().round(2)
print("\nCluster Summary:\n", cluster_summary)

# Step 5: Recommendations based on cluster profiles
recommendations = {
    0: "Younger customers with lower income and moderate spend - Target with budget promotions and entry-level products.",
    1: "Middle-aged, medium-high income, high spend - Offer premium products and loyalty programs.",
    2: "Older, high income, high spend - Provide exclusive memberships and luxury product offers.",
    3: "Older, high income, low spend - Engage to increase spending, collect feedback, and offer customized campaigns."
}

print("\nMarketing Recommendations:")
for cluster, rec in recommendations.items():
    print(f"Cluster {cluster}: {rec}")
