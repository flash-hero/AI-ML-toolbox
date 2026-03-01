from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import product
import pandas as pd

class OPTICSClustering:
    """
    OPTICS Clustering.
    (Ordering Points To Identify the Clustering Structure)
    
    How it works:
    Similar to DBSCAN but better at handling clusters of DIFFERENT densities.
    It builds a reachability plot that orders points by density.
    
    Why use it?
    - Detects clusters of varying density (DBSCAN struggles with this).
    - Can handle noise/outliers.
    """

    def __init__(self, data):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.best_params = None
        self.model = None
        self.labels = None

    def _optimize_hyperparameters(self, param_grid=None, n_iter=10):
        """
        Random Search for best hyperparameters.
        """
        print("Optimizing OPTICS parameters (Random Search)...")
        if param_grid is None:
            param_grid = {
                'min_samples': [5, 10, 20],      # Minimum points to be considered a core point
                'max_eps': [np.inf, 0.5, 1.0, 5.0], # Maximum distance (inf is standard for OPTICS)
                'metric': ['euclidean', 'minkowski', 'cosine'],
                'cluster_method': ['xi', 'dbscan'], # 'xi' is the newer method, 'dbscan' gives DBSCAN-like results
                'xi': [0.01, 0.05] # Only used if cluster_method='xi'
            }

        keys, values = zip(*param_grid.items())
        all_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        # Try a random subset of combinations to save time
        sampled_combinations = random.sample(all_combinations, min(n_iter, len(all_combinations)))

        best_score = -1
        best_config = None

        for params in sampled_combinations:
            try:
                model = OPTICS(**params)
                labels = model.fit_predict(self.data)

                # Check validity
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

                if n_clusters > 1:
                    mask = labels != -1
                    if np.sum(mask) > 0:
                        score = silhouette_score(self.data[mask], labels[mask])
                        
                        # print(f"Test {params} -> Clusters: {n_clusters} | Silhouette: {score:.4f}")

                        if score > best_score:
                            best_score = score
                            best_config = params
            except Exception as e:
                pass

        self.best_params = best_config
        if best_config:
            print(f"✔ Best Config: {best_config}")
            print(f"✔ Best Silhouette Score: {best_score:.4f}")
        else:
            print("⚠ No good clustering found. Using default parameters.")
            self.best_params = {'min_samples': 5}

    def _fit_model(self):
        print("Training Final OPTICS Model...")
        self.model = OPTICS(**self.best_params)
        self.labels = self.model.fit_predict(self.data)

    def _describe_clusters(self):
        print("\nCluster Description:")
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            if label == -1:
                count = np.sum(self.labels == -1)
                print(f"Noise (Outliers) : {count} points")
            else:
                count = np.sum(self.labels == label)
                percent = 100 * count / len(self.data)
                print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _evaluate_performance(self):
        if len(set(self.labels)) > 1:
            mask = self.labels != -1
            if len(set(self.labels[mask])) > 1:
                silhouette = silhouette_score(self.data[mask], self.labels[mask])
                print(f"✔ Final Silhouette Score: {silhouette:.4f}")
            else:
                print("⚠ Mostly noise, cannot evaluate properly.")
        else:
            print("⚠ Only 1 cluster (or only noise) found.")

    def _visualize_clusters(self):
        # Use PCA for visualization if dims > 2
        dims = self.data.shape[1]
        
        if dims > 2:
            print("Reducing dimensions with PCA for visualization...")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(self.data)
            title_suffix = "(PCA Projection)"
        else:
            coords = self.data
            title_suffix = "(2D)"

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=self.labels, cmap='tab10', s=40)
        plt.title(f"OPTICS Clustering {title_suffix}")
        
        # Create a legend
        unique_labels = np.unique(self.labels)
        # We can't easily add a legend with plt.scatter c=array, so we use colorbar or loop
        plt.colorbar(scatter, label="Cluster Label (-1 is Noise)")
        plt.grid(True)
        plt.show()

    def run_optics_clustering(self, n_iter=10):
        print("=== Running OPTICS Clustering ===")
        self._optimize_hyperparameters(n_iter=n_iter)
        self._fit_model()
        self._describe_clusters()
        self._evaluate_performance()
        self._visualize_clusters()
