from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

class KMeansClustering:
    """
    K-Means Clustering.
    
    How it works:
    1. Randomly picks 'k' centers (centroids).
    2. Assigns every point to the nearest center.
    3. Moves the center to the average of the points assigned to it.
    4. Repeats until centers stop moving.
    
    Why use it?
    - Fast and simple.
    - Works very well for spherical, evenly sized clusters.
    """

    def __init__(self, data, max_clusters=10):
        self.data = data
        self.max_clusters = max_clusters
        self.best_k = None
        self.model = None
        self.labels = None
        self.centroids = None

    def _find_optimal_k(self):
        """
        Calculates Silhouette Score for k=2 to max_clusters to find best k.
        """
        print("Optimizing k (Number of Clusters)...")
        best_score = -1
        silhouette_scores = []

        for k in range(2, self.max_clusters + 1):
            model = KMeans(n_clusters=k, n_init=30, random_state=42)
            model.fit(self.data)
            score = silhouette_score(self.data, model.labels_)
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                self.best_k = k

        self._plot_silhouette(silhouette_scores)
        print(f"✔ Best k: {self.best_k} | Silhouette Score: {best_score:.4f}")

    def _fit_model(self):
        print(f"Training K-Means with k={self.best_k}...")
        self.model = KMeans(n_clusters=self.best_k, n_init=30, random_state=42)
        self.model.fit(self.data)
        self.labels = self.model.labels_
        self.centroids = self.model.cluster_centers_

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        data = self.data.values if isinstance(self.data, pd.DataFrame) else self.data
        labels = self.labels
        centroids = self.centroids

        if dims == 2:
            X1, X2 = data[:, 0], data[:, 1]
            distances = euclidean_distances(data, centroids)
            radius = [np.max(distances[labels == i, i]) for i in range(len(centroids))]

            plt.figure(figsize=(10, 6))
            plt.scatter(X1, X2, c=labels, s=40, cmap='viridis', alpha=0.6)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')

            for i, centroid in enumerate(centroids):
                circle = plt.Circle(centroid, radius[i], color='black', fill=False, alpha=0.3)
                plt.gca().add_patch(circle)

            plt.title("K-Means Clustering (2D)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.show()

        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=40, cmap='viridis')
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=100, label='Centroids')
            ax.set_title("K-Means Clustering (3D)")
            plt.legend()
            plt.show()
        else:
            print("Visualization only available for 2D or 3D data.")

    def _plot_silhouette(self, scores):
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, self.max_clusters + 1), scores, marker='o')
        plt.title("Silhouette Scores")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    def run_k_means_clustering(self):
        print("=== Running K-Means Clustering ===")
        self._find_optimal_k()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
