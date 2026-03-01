import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

class AgglomerativeBirchClustering:
    """
    Agglomerative + BIRCH Clustering.
    
    How it works:
    A hybrid two-step approach:
    1. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies): 
       Quickly compresses the large dataset into tiny, manageable "sub-clusters".
    2. Agglomerative Clustering: 
       Groups those sub-clusters into the final clusters.
    
    Why use it?
    - BIRCH is super fast and memory efficient for large datasets.
    - Agglomerative Clustering creates high-quality hierarchy but is slow.
    - Combining them gives the best of both worlds: Speed + Quality.
    """

    def __init__(self, data, max_clusters=10, birch_threshold=0.5, branching_factor=50, linkage='ward'):
        self.data = data
        self.max_clusters = max_clusters
        self.best_k = None
        self.model = None
        self.labels = None
        self.centroids = None
        self.birch_threshold = birch_threshold
        self.branching_factor = branching_factor
        self.linkage = linkage

    def _find_optimal_k(self):
        """
        Finds the best number of clusters (k) by checking Silhouette Scores.
        """
        print("Finding optimal number of clusters...")
        best_score = -1
        silhouette_scores = []

        # First stage: BIRCH pre-clustering (Compressing data)
        # n_clusters=None means Birch purely compresses data into subclusters, not final clusters yet.
        birch = Birch(threshold=self.birch_threshold,
                     branching_factor=self.branching_factor,
                     n_clusters=None)
        birch.fit(self.data)
        
        # Second stage: Agglomerative on BIRCH subclusters
        # We try different k values
        for k in range(2, self.max_clusters + 1):
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels = model.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                self.best_k = k

        self._plot_silhouette(silhouette_scores)
        print(f"Best k : {self.best_k} | Silhouette Score : {best_score:.4f}")

    def _fit_model(self):
        """
        Fits the final model using the best k.
        """
        # Compress data with BIRCH
        birch = Birch(threshold=self.birch_threshold,
                     branching_factor=self.branching_factor,
                     n_clusters=None)
        birch.fit(self.data)

        # Cluster the result with Agglomerative Clustering
        self.model = AgglomerativeClustering(n_clusters=self.best_k, linkage=self.linkage)
        self.labels = self.model.fit_predict(self.data)

        # Calculate centroids manually (Agglomerative doesn't produce centroids by default)
        self.centroids = np.array([self.data[self.labels == i].mean(axis=0)
                                 for i in range(self.best_k)])

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
            
            # Draw circles around clusters
            for i, centroid in enumerate(centroids):
                circle = plt.Circle(centroid, radius[i], color='black', fill=False, alpha=0.3)
                plt.gca().add_patch(circle)
                
            plt.title("Agglomerative-BIRCH Clustering (2D)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.show()
            
        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=40, cmap='viridis')
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=100, label='Centroids', marker='X')
            ax.set_title("Agglomerative-BIRCH Clustering (3D)")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("X3")
            plt.legend()
            plt.show()
        else:
            print("Visualization available only for 2D or 3D data.")

    def _plot_silhouette(self, scores):
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, self.max_clusters + 1), scores, marker='o')
        plt.title("Silhouette Scores by k")
        plt.xlabel("k (Number of Clusters)")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    def run_agglomerative_birch_clustering(self):
        print("=== Running Agglomerative-BIRCH Clustering ===")
        self._find_optimal_k()
        self._fit_model()
        self._describe_clusters()
        self._visualize()