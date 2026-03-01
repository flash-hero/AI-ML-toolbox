import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class DensityPeaksClustering:
    """
    Density Peaks Clustering (DPC).
    
    How it works:
    Based on the idea that cluster centers satisfy two conditions:
    1. High local density (surrounded by many points).
    2. Large distance from any point with even higher density.
    
    Why use it?
    - Can find non-spherical clusters.
    - Identifies the number of clusters automatically via a decision graph.
    """

    def __init__(self, data, dc_range=[1.0, 2.0, 3.0, 4.0, 5.0], max_k_range=range(2, 11)):
        self.data = data.values if hasattr(data, 'values') else data
        self.dc_range = dc_range
        self.max_k_range = max_k_range
        self.best_dc = None
        self.best_k = None
        self.best_score = -1
        self.results = {}

    def _compute_distances(self):
        return squareform(pdist(self.data, 'euclidean'))

    def _compute_density(self, distances, dc):
        """Calculates local density (rho) for each point."""
        # Gaussian kernel
        return np.sum(np.exp(-(distances / dc) ** 2), axis=1) - 1

    def _compute_delta(self, distances, density):
        """Calculates minimum distance to a higher density point (delta)."""
        n = len(density)
        delta = np.zeros(n)
        nearest_higher = np.zeros(n, dtype=int)
        
        # Sort points by density (descending)
        sorted_indices = np.argsort(-density)
        
        # Point with highest density has delta = max distance to any point
        delta[sorted_indices[0]] = np.max(distances[sorted_indices[0]])
        nearest_higher[sorted_indices[0]] = -1
        
        for i in range(1, n):
            current_idx = sorted_indices[i]
            higher_indices = sorted_indices[:i]
            
            # Find closest point among those with higher density
            dist_to_higher = distances[current_idx, higher_indices]
            min_dist = np.min(dist_to_higher)
            
            delta[current_idx] = min_dist
            nearest_higher[current_idx] = higher_indices[np.argmin(dist_to_higher)]
            
        return delta, nearest_higher

    def _assign_clusters(self, density, delta, nearest_higher, distances, k):
        # Gamma = Density * Delta. High gamma = likely Cluster Center.
        gamma = density * delta
        centers = np.argsort(-gamma)[:k]
        
        labels = -np.ones(len(density), dtype=int)
        
        # Assign unique label to centers
        for cluster_id, center in enumerate(centers):
            labels[center] = cluster_id
            
        # Assign everyone else to the same cluster as their nearest neighbor of higher density
        sorted_indices = np.argsort(-density)
        for idx in sorted_indices:
            if labels[idx] == -1:
                labels[idx] = labels[nearest_higher[idx]]
                
        return labels, centers

    def _plot_heatmap(self, scores):
        df = pd.DataFrame(scores, columns=["dc", "k", "score"])
        df["dc"] = df["dc"].round(2)
        pivot = df.pivot(index="dc", columns="k", values="score")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Silhouette Score (Hyperparameter tuning)")
        plt.xlabel("k (Number of clusters)")
        plt.ylabel("dc (Cutoff distance %)")
        plt.show()

    def _describe_clusters(self, labels):
        print("\nCluster Description:")
        for label in np.unique(labels):
            count = np.sum(labels == label)
            percent = 100 * count / len(labels)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self, labels, centers):
        dims = self.data.shape[1]
        if dims == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
            plt.scatter(self.data[centers, 0], self.data[centers, 1], c='red', marker='x', s=200, label='Centers')
            plt.title('Density Peaks Clustering (2D)')
            plt.legend()
            plt.show()
        elif dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=labels, cmap='viridis', s=50)
            ax.scatter(self.data[centers, 0], self.data[centers, 1], self.data[centers, 2], 
                       c='red', marker='x', s=200, label='Centers')
            ax.set_title('Density Peaks Clustering (3D)')
            ax.legend()
            plt.show()

    def run_density_peaks_clustering(self):
        print("=== Running Density Peaks Clustering ===")
        print("Optimizing 'dc' (kernel size) and 'k'...")
        
        distances = self._compute_distances()
        scores = []

        for dc in self.dc_range:
            # dc is percentile (e.g., 2% of avg distance)
            dc_val = np.percentile(distances, dc)
            density = self._compute_density(distances, dc_val)
            delta, nearest_higher = self._compute_delta(distances, density)
            
            for k in self.max_k_range:
                try:
                    labels, centers = self._assign_clusters(density, delta, nearest_higher, distances, k)
                    score = silhouette_score(self.data, labels)
                    scores.append((dc, k, score))
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_dc = dc
                        self.best_k = k
                        self.results = {
                            'labels': labels,
                            'centers': centers,
                            'density': density,
                            'delta': delta,
                            'dc': dc_val,
                            'gamma': density * delta,
                            'nearest_higher': nearest_higher
                        }
                except:
                    continue

        self._plot_heatmap(scores)
        print(f"\n✔ Best Fit: dc={self.best_dc}%, k={self.best_k} | Silhouette Score: {self.best_score:.4f}")
        
        if self.results:
            self._describe_clusters(self.results['labels'])
            self._visualize(self.results['labels'], self.results['centers'])
        else:
            print("Failed to find a valid clustering.")

        return self.results
