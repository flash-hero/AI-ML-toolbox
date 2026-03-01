import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

class BIRCHClustering:
    """
    BIRCH Clustering.
    
    How it works:
    (Balanced Iterative Reducing and Clustering using Hierarchies)
    It builds a tree structure (CF Tree) to process data points incrementally.
    
    Why use it?
    - Designed specifically for VERY large datasets.
    - Can cluster data with a single pass (very fast).
    - Memory efficient.
    """

    def __init__(self, data, max_clusters=10, threshold=0.5, branching_factor=50):
        self.data = data
         # Ensure data is appropriately formatted
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            
        self.max_clusters = max_clusters
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.best_k = None
        self.model = None
        self.labels = None

    def _find_optimal_k(self):
        """
        Test different numbers of clusters 'k' to find the best one.
        """
        print("Finding optimal number of clusters for BIRCH...")
        best_score = -1
        silhouette_scores = []
        
        # We start from k=2 because k=1 makes no sense for silhouette score
        for k in range(2, self.max_clusters + 1):
            try:
                model = Birch(n_clusters=k, threshold=self.threshold, branching_factor=self.branching_factor)
                labels = model.fit_predict(self.data)
                
                # Check if we actually got > 1 cluster
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(self.data, labels)
                else:
                    score = -1
                    
                silhouette_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    self.best_k = k
            except Exception as e:
                # print(f"Error for k={k}: {e}")
                silhouette_scores.append(-1)
                continue
                
        # If no valid k found, default to 3
        if self.best_k is None:
            self.best_k = 3
            print("Could not determine optimal k. Defaulting to 3.")
        else:
            print(f"Best k found: {self.best_k} | Silhouette Score: {best_score:.4f}")

        # Plot Silhouette Scores
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, self.max_clusters + 1), silhouette_scores, marker='o')
        plt.title("Silhouette Score vs Number of Clusters")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    def _fit_model(self):
        """
        Train the final model with best k.
        """
        print(f"Training BIRCH with k={self.best_k}...")
        self.model = Birch(n_clusters=self.best_k, threshold=self.threshold, branching_factor=self.branching_factor)
        self.labels = self.model.fit_predict(self.data)

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        
        if dims == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', s=50, alpha=0.8)
            plt.title(f"BIRCH Clustering (k={self.best_k})")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.colorbar(label='Cluster')
            plt.show()
        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels, cmap='viridis', s=50)
            ax.set_title(f"BIRCH Clustering (k={self.best_k})")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("X3")
            plt.colorbar(img, label='Cluster')
            plt.show()
        else:
            print("Visualization available only for 2D or 3D data.")

    def run_birch_clustering(self):
        print("=== Running BIRCH Clustering ===")
        self._find_optimal_k()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
