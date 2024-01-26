import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Reduce the dimensionality of the data to 2 dimensions for plotting
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# Create subplots for each value of k
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(left=0.02, right=0.98, wspace=0.2)

# Perform k-means clustering with k=3, 4, and 5
for k, ax in zip([3, 4, 5], axes):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Plot the data points with different colors for each cluster
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i in range(k):
        ax.scatter(X_r[labels == i, 0], X_r[labels == i, 1],
                   label=f'Cluster {i + 1}', c=colors[i])

    ax.set_title(f'k = {k}')
    ax.legend()

plt.show()
