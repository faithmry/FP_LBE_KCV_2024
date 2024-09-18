import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
df = pd.read_csv('./DataStuntingIndonesia.csv', delimiter = ';')

# Remove the 5th column
df = df.drop(columns=['Unnamed: 5'])

df

df.info()

df.describe()


# Convert comma-separated decimals to periods and then to float
for year in ['2020', '2021', '2022', '2023']:
    df[year] = df[year].astype(str).str.replace(',', '.').astype(float)

# Select features for clustering
X = df[['2020', '2021', '2022', '2023']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 2. Choose the number of clusters (Elbow method)
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()


# Next Calculate silhouette scores for different values of k
from sklearn.metrics import silhouette_score
silhouette_scores = []
for k in range(2, 11):  # Start from 2 clusters, as silhouette score is undefined for k=1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Optimal k')
plt.show()


# 3. Apply k-means clustering
# Let's say we choose k=3 based on the elbow curve
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Pairplot after clustering
sns.pairplot(df, hue='Cluster', vars=['2020', '2021', '2022', '2023'])
plt.show()


# 4. Analyze the results
for cluster in df['Cluster'].unique():
    print(f"\nCluster {cluster}:")
    cluster_data = df[df['Cluster'] == cluster]
    print(cluster_data[['Provinsi', '2020', '2021', '2022', '2023']])
    print("\nMean values:")
    print(cluster_data[['2020', '2021', '2022', '2023']].mean())
    print("-" * 50)


# Visualize the clusters (using first two features)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X['2022'], X['2023'], c=df['Cluster'], cmap='viridis')
plt.xlabel('2022')
plt.ylabel('2023')
plt.title('K-means Clustering Results')
plt.colorbar(scatter, label='Cluster')

# Annotate with 'Provinsi'
for i, txt in enumerate(df['Provinsi']):
    plt.annotate(txt, (X.iloc[i]['2022'], X.iloc[i]['2023']), fontsize=8)

plt.tight_layout()
plt.show()