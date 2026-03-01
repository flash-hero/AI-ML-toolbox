import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class HDBSCANClustering:
    """
    HDBSCAN Clustering.
    
    How it works:
    Hierarchy-based Density-Based Spatial Clustering.
    It extends DBSCAN by converting it into a hierarchical clustering algorithm, then extracting a flat clustering based on the stability of clusters.
    
    Why use it?
    - Best-in-class for density-based clustering.
    - More robust than standard DBSCAN (less sensitive to 'eps' parameter).
    """

    def __init__(self, data):
        self.data = data
        self.model = None
        self.labels = None
        self.best_params = None
        self.silhouette_avg = None

    def _optimize_parameters(self):
        """
        Grid Search for best parameters.
        """
        print("Optimizing HDBSCAN parameters...")

        min_cluster_sizes = [5, 10, 15, 20, 25, 30]
        min_samples = [5, 10, 15, 20]

        best_score = -1
        best_params = {}
        
        results = []

        for min_cluster_size in min_cluster_sizes:
            for min_sample in min_samples:
                if min_sample > min_cluster_size:
                    continue

                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_sample,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )

                model.fit(self.data)
                labels = model.labels_

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                result = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_sample,
                    'n_clusters': n_clusters,
                    'noise_points': list(labels).count(-1),
                    'silhouette': None
                }

                if n_clusters >= 2:
                    mask = labels != -1
                    if sum(mask) > 0:
                        silhouette_avg = silhouette_score(self.data[mask], labels[mask])
                        result['silhouette'] = silhouette_avg

                        if silhouette_avg > best_score:
                            best_score = silhouette_avg
                            best_params = {
                                'min_cluster_size': min_cluster_size,
                                'min_samples': min_sample
                            }

                results.append(result)

        if best_score == -1:
            print("\n⚠ No valid clustering found. Using default parameters.")
            self.best_params = {'min_cluster_size': 5, 'min_samples': 5}
        else:
            print(f"\n✔ Best parameters: {best_params}")
            print(f"✔ Best Silhouette Score (ignoring noise): {best_score:.4f}")
            self.best_params = best_params
            self.silhouette_avg = best_score

    def _fit_model(self):
        print("Training Final HDBSCAN Model...")
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.best_params['min_cluster_size'],
            min_samples=self.best_params['min_samples'],
            metric='euclidean',
            cluster_selection_method='eom'
        )
        self.model.fit(self.data)
        self.labels = self.model.labels_

    def _describe_clusters(self):
        unique_labels = set(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.labels).count(-1)

        print(f"\nNumber of clusters: {n_clusters}")
        print(f"Noise points: {n_noise} ({n_noise/len(self.labels)*100:.2f}%)")

        print("\nCluster Description:")
        for label in sorted(unique_labels):
            if label == -1:
                continue
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label}: {count} points ({percent:.2f}%)")

    def _visualize_clusters(self):
        dims = self.data.shape[1]

        if dims == 2:
            X_2d = self.data
            title_suffix = "(Original 2D)"
        else:
            print("\nReducing dimensions with t-SNE for visualization...")
            X_2d = TSNE(n_components=2, random_state=42).fit_transform(self.data)
            title_suffix = "(t-SNE Projection)"

        plt.figure(figsize=(10, 8))
        unique_labels = set(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        palette = sns.color_palette("husl", n_clusters)

        # Plot noise
        noise_mask = self.labels == -1
        if np.any(noise_mask):
            plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], c="lightgrey", marker=".", s=20, alpha=0.3, label="Noise")

        # Plot clusters
        color_idx = 0
        for label in sorted(unique_labels):
            if label == -1: continue
            
            mask = self.labels == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[palette[color_idx]], marker="o", s=60, alpha=0.8, label=f"Cluster {label}")
            color_idx += 1

        plt.title(f"HDBSCAN Clustering {title_suffix}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_hdbscan_clustering(self):
        print("=== Running HDBSCAN Clustering ===")
        self._optimize_parameters()
        self._fit_model()
        self._describe_clusters()
        self._visualize_clusters()