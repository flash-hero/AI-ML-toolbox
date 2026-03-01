from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class MeanShiftClustering:
    """
    Mean Shift Clustering.
    
    How it works:
    Imagine the data points as a density map (like mountains).
    It puts a window on each point and moves the window 'uphill' towards higher density.
    Points that end up at the same peak belong to the same cluster.
    
    Why use it?
    - No need to specify 'k' (number of clusters). It finds it automatically.
    - Robust to irregular shapes.
    """

    def __init__(self, data, quantile=0.2):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.quantile = quantile
        self.bandwidth = None
        self.model = None
        self.labels = None
        self.centroids = None

    def _estimate_bandwidth(self):
        print(f"Estimating bandwidth (quantile={self.quantile})...")
        try:
            self.bandwidth = estimate_bandwidth(self.data, quantile=self.quantile, n_samples=min(len(self.data), 1000))
            print(f"✔ Bandwidth: {self.bandwidth:.4f}")
        except Exception as e:
            print(f"Bandwidth estimation failed: {e}. Using default.")
            self.bandwidth = None

    def _fit_model(self):
        print("Training Mean Shift...")
        if self.bandwidth:
            self.model = MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
        else:
            self.model = MeanShift(bin_seeding=True) # Let sklearn guess
            
        self.model.fit(self.data)
        self.labels = self.model.labels_
        self.centroids = self.model.cluster_centers_
        
        n_clusters = len(np.unique(self.labels))
        print(f"✔ Found {n_clusters} clusters.")
        
        if n_clusters > 1:
            try:
                score = silhouette_score(self.data, self.labels)
                print(f"✔ Silhouette Score: {score:.4f}")
            except:
                pass

    def _describe_clusters(self):
        print("\nCluster Description:")
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        
        if dims == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis', s=40, alpha=0.7)
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
            plt.title(f"Mean Shift Clustering (2D) - {len(self.centroids)} Clusters")
            plt.legend()
            plt.show()
        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels, cmap='viridis', s=40)
            ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], c='black', marker='X', s=100, label='Centroids')
            ax.set_title("Mean Shift Clustering (3D)")
            plt.legend()
            plt.show()
        else:
             print("Visualization available only for 2D or 3D data.")

    def run_mean_shift_clustering(self):
        print("=== Running Mean Shift Clustering ===")
        self._estimate_bandwidth()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
