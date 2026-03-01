from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class DivisiveHierarchicalClustering:
    """
    Divisive Hierarchical Clustering (using Bisecting K-Means).
    
    How it works:
    "Top-down" approach (opposite of Agglomerative).
    1. Starts with ALL data in one big cluster.
    2. Repeatedly splits the "worst" cluster into two smaller clusters (using K-Means with k=2).
    3. Stops when it reaches 'k' clusters.
    
    Why use it?
    - More efficient than Agglomerative for large datasets ($O(n)$ vs $O(n^2)$).
    - Produces a hierarchy of clusters.
    """

    def __init__(self, data, max_clusters=10):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.max_clusters = max_clusters
        self.best_k = None
        self.best_score = -1
        self.model = None
        self.labels = None
        self.centroids = None

    def _find_optimal_k(self):
        """
        Tests differents values of k to find the best Silhouette score.
        """
        print("Finding optimal k using Bisecting K-Means...")
        silhouette_scores = []
        
        for k in range(2, self.max_clusters + 1):
            try:
                # BisectingKMeans introduced in sklearn 1.1
                model = BisectingKMeans(n_clusters=k, random_state=42)
                labels = model.fit_predict(self.data)
                
                score = silhouette_score(self.data, labels)
                silhouette_scores.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_k = k
            except Exception as e:
                print(f"Error for k={k}: {e}")
                continue
        
        if self.best_k is None:
            self.best_k = 3 # Default
            
        print(f"✔ Best k: {self.best_k} | Silhouette Score: {self.best_score:.4f}")
        
        # Plot Scores
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
        plt.title("Silhouette Score vs k (Divisive)")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    def _fit_model(self):
        print(f"Training Divisive Clustering with k={self.best_k}...")
        self.model = BisectingKMeans(n_clusters=self.best_k, random_state=42)
        self.labels = self.model.fit_predict(self.data)
        self.centroids = self.model.cluster_centers_

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label}: {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        
        if dims == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', s=40, alpha=0.7)
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
            plt.title("Divisive Hierarchical Clustering (2D)")
            plt.legend()
            plt.show()
        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels, cmap='viridis', s=40)
            ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], c='black', marker='X', s=100, label='Centroids')
            ax.set_title("Divisive Hierarchical Clustering (3D)")
            plt.legend()
            plt.show()
        else:
            print("Visualization available only for 2D or 3D data.")

    def run_divisive_hierarchical_clustering_clustering(self):
        print("=== Running Divisive Hierarchical Clustering ===")
        try:
             self._find_optimal_k()
             self._fit_model()
             self._describe_clusters()
             self._visualize()
        except ImportError:
             print("Error: Divisive Hierarchical requires 'BisectingKMeans' (sklearn version >= 1.1).")
        except Exception as e:
             print(f"An error occurred: {e}")
