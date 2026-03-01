from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class DBSCANClustering:
    """
    DBSCAN Clustering.
    
    How it works:
    (Density-Based Spatial Clustering of Applications with Noise)
    It groups together points that are close to each other (high density) 
    and marks points that are far away from any group as "Noise" (outliers).
    
    Why use it?
    - Can find clusters of ANY shape (not just circles like K-Means).
    - Automatically handles outliers.
    - You don't need to specify 'k'.
    """

    def __init__(self, data):
        self.data = data
        self.model = None
        self.labels = None
        self.best_params = None
        self.best_score = -1

    def _optimize_hyperparameters(self):
        """
        Grid Search to find best 'eps' (distance) and 'min_samples' (density threshold).
        """
        print("Optimizing DBSCAN hyperparameters...")
        param_grid = ParameterGrid({
            'eps': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],        # Radius to look for neighbors
            'min_samples': [3, 5, 10, 20],                # Min points to form a cluster
            'metric': ['euclidean', 'manhattan'],
            'algorithm': ['auto']
        })

        for params in param_grid:
            try:
                model = DBSCAN(**params)
                labels = model.fit_predict(self.data)
                
                # Check that we have valid clusters (not just noise + 1 cluster, or all noise)
                # Label -1 is noise.
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                
                # We need at least 2 real clusters to measure silhouette, or 1 cluster + noise?
                # Usually Silhouette is not great for DBSCAN density, but it's a proxy.
                if n_clusters < 2:
                    continue
                    
                score = silhouette_score(self.data, labels)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self.model = model
                    self.labels = labels
                    
            except Exception as e:
                continue

        if self.best_params:
            print(f"✔ Best parameters: {self.best_params}")
            print(f"✔ Best Silhouette Score: {self.best_score:.4f}")
        else:
            print("⚠ No valid clusters found. Try scaling your data or changing parameter ranges.")
            # Default fallback if nothing works
            self.model = DBSCAN(eps=0.5, min_samples=5)
            self.labels = self.model.fit_predict(self.data)

    def _fit_model(self):
        if self.model is None:
             pass # Handled in optimization
        else:
             # Already fitted
             pass

    def _describe_clusters(self):
        if self.labels is None:
            return
        unique_labels = set(self.labels)
        print("\nCluster Description:")
        for label in sorted(unique_labels):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            label_name = "NOISE (-1)" if label == -1 else f"Cluster {label}"
            print(f"{label_name} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        data = self.data.values if isinstance(self.data, pd.DataFrame) else self.data
        labels = self.labels

        if dims == 2:
            plt.figure(figsize=(10, 6))
            unique_labels = set(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]  # Black for noise
                    marker = 'x'
                    label_text = "Noise"
                else:
                    marker = 'o'
                    label_text = f"Cluster {k}"
                
                class_member_mask = (labels == k)
                xy = data[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_text, s=50 if k!=-1 else 20, marker=marker)

            plt.title("DBSCAN Clustering (2D)")
            plt.legend()
            plt.grid(True)
            plt.show()

        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10', s=40)
            ax.set_title("DBSCAN Clustering (3D)")
            plt.show()
        else:
            print("Visualization only for 2D or 3D.")

    def run_dbscan_clustering(self):
        print("=== Running DBSCAN Clustering ===")
        self._optimize_hyperparameters()
        self._describe_clusters()
        self._visualize()
