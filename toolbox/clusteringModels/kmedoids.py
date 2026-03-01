from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

class KMedoidsClustering:
    """
    K-Medoids Clustering (PAM - Partitioning Around Medoids).
    
    How it works:
    Similar to K-Means, BUT the center of a cluster MUST be one of the actual data points (called a Medoid).
    K-Means uses "Centroids" (average of points, which might not be a real data point).
    
    Why use it?
    - More robust to outliers than K-Means.
    - Can work with arbitrary distance matrices (not just Euclidean).
    """

    def __init__(self, data, max_clusters=10):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.max_clusters = max_clusters
        self.best_k = None
        self.model = None
        self.labels = None
        self.medoids = None
        self.silhouette_scores = []

    def _find_optimal_k(self):
        print("Optimizing k for K-Medoids...")
        best_score = -1

        for k in range(2, self.max_clusters + 1):
            model = KMedoids(n_clusters=k, random_state=42)
            model.fit(self.data)
            score = silhouette_score(self.data, model.labels_)
            self.silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                self.best_k = k

        print(f"✔ Best k: {self.best_k} | Silhouette Score: {best_score:.4f}")
        self._plot_silhouette()

    def _fit_model(self):
        if self.best_k is None:
            raise ValueError("You must run _find_optimal_k first.")
        
        print(f"Training K-Medoids with k={self.best_k}...")
        self.model = KMedoids(n_clusters=self.best_k, random_state=42)
        self.model.fit(self.data)
        self.labels = self.model.labels_
        # cluster_centers_ in KMedoids are the medoids themselves (coordinates)
        self.medoids = self.model.cluster_centers_

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        data = self.data
        labels = self.labels
        medoids = self.medoids

        if dims == 2:
            X1, X2 = data[:, 0], data[:, 1]
            distances = euclidean_distances(data, medoids)
            radius = [np.max(distances[labels == i, i]) for i in range(len(medoids))]

            plt.figure(figsize=(10, 8))
            plt.scatter(X1, X2, c=labels, s=40, cmap='viridis', alpha=0.6)
            plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='x', s=100, label='Medoids')
            for i, medoid in enumerate(medoids):
                circle = plt.Circle(medoid, radius[i], color='black', fill=False, alpha=0.3)
                plt.gca().add_patch(circle)

            plt.title("K-Medoids Clustering (2D)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.show()

        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=40, cmap='viridis')
            ax.scatter(medoids[:, 0], medoids[:, 1], medoids[:, 2], c='black', s=100, label='Medoids')
            ax.set_title("K-Medoids Clustering (3D)")
            plt.legend()
            plt.show()

        else:
            print("Visualization only available for 2D or 3D data.")

    def _plot_silhouette(self):
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, self.max_clusters + 1), self.silhouette_scores, marker='o')
        plt.title("Silhouette Scores")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    def run_k_medoids_clustering(self):
        print("=== Running K-Medoids Clustering ===")
        self._find_optimal_k()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
