from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MiniBatchKMeansClustering:
    """
    Mini-Batch K-Means Clustering.
    
    How it works:
    Just like K-Means, but instead of using ALL data to update centroids at every step, 
    it uses small random batches (e.g., 256 points).
    
    Why use it?
    - Much FASTER than standard K-Means for huge datasets.
    - Uses less memory.
    - Slightly less accurate than full K-Means, but usually good enough.
    """

    def __init__(self, data, max_clusters=10, batch_sizes=[16, 32, 64, 128, 256]):
        self.data = data
        self.max_clusters = max_clusters
        self.batch_sizes = batch_sizes
        self.best_k = None
        self.best_batch_size = None
        self.model = None
        self.labels = None
        self.centroids = None

    def _find_optimal_k_and_batch(self):
        """
        Grid Search to find best k and batch_size combination.
        """
        print("Optimizing k and batch_size...")
        best_score = -1
        results = []

        for k in range(2, self.max_clusters + 1):
            for batch_size in self.batch_sizes:
                model = MiniBatchKMeans(
                    n_clusters=k, 
                    batch_size=batch_size, 
                    random_state=42, 
                    n_init=3
                )
                model.fit(self.data)
                score = silhouette_score(self.data, model.labels_)
                
                results.append((k, batch_size, score))

                if score > best_score:
                    best_score = score
                    self.best_k = k
                    self.best_batch_size = batch_size

        self._plot_silhouette(results)
        print(f"✔ Best Config: k={self.best_k}, batch_size={self.best_batch_size} | Silhouette Score: {best_score:.4f}")

    def _fit_model(self):
        print(f"Training Mini-Batch K-Means (k={self.best_k})...")
        self.model = MiniBatchKMeans(
            n_clusters=self.best_k, 
            batch_size=self.best_batch_size, 
            random_state=42, 
            n_init=3
        )
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
            radius = [np.max(distances[labels == i, i]) if np.any(labels==i) else 0 for i in range(len(centroids))]
            
            plt.figure(figsize=(10,6))
            plt.scatter(X1, X2, c=labels, s=40, cmap='viridis', alpha=0.6)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
            
            for i, centroid in enumerate(centroids):
                circle = plt.Circle(centroid, radius[i], color='black', fill=False, alpha=0.3)
                plt.gca().add_patch(circle)
                
            plt.title("Mini-Batch K-Means (2D)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.show()
        elif dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=40, cmap='viridis')
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=100, label='Centroids')
            ax.set_title("Mini-Batch K-Means (3D)")
            plt.legend()
            plt.show()
        else:
            print("Visualization only available for 2D or 3D data.")

    def _plot_silhouette(self, results):
        import seaborn as sns
        df = pd.DataFrame(results, columns=["k", "batch_size", "silhouette"])
        pivot = df.pivot(index="batch_size", columns="k", values="silhouette")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Silhouette Score (Batch Size vs k)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Batch Size")
        plt.show()

    def run_minibatch_kmeans_clustering(self):
        print("=== Running Mini-Batch K-Means Clustering ===")
        self._find_optimal_k_and_batch()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
