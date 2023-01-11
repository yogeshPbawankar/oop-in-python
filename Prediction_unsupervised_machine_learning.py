from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Use the elbow method to find the optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit the K-Means model with the optimum number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

# Visualize the clusters using a scatter plot
plt.scatter(X[pred_y == 0, 0], X[pred_y == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[pred_y == 1, 0], X[pred_y == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolor')
plt.scatter(X[pred_y == 2, 0], X[pred_y == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()
