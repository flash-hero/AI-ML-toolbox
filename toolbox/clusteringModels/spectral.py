from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SpectralClusteringModel:
    """
    Spectral Clustering.
    
    How it works:
    Uses the "spectrum" (eigenvalues) of the similarity matrix of the data to reduce dimensions, 
    then applies K-Means in this lower-dimensional space.
    
    Why use it?
    - Can find clusters with complex shapes (like intertwined spirals) that K-Means fails on.
    - Graph-based approach.
    """

    def __init__(self, data, max_clusters=10, gamma_range=None):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.max_clusters = max_clusters
        # Gamma controls the breadth of the RBF kernel (similarity function)
        self.gamma_range = gamma_range if gamma_range else [0.01, 0.1, 1.0, 10.0] 
        self.best_k = None
        self.best_gamma = None
        self.model = None
        self.labels = None
        self.centroids = None

    def _find_optimal_k_gamma(self):
        """
        Grid Search for optimal k and gamma.
        """
        print("Optimizing k and gamma (Similarity)...")
        best_score = -1
        best_config = (None, None)
        scores = []

        for gamma in self.gamma_range:
            # Precompute affinity matrix to be faster
            affinity_matrix = rbf_kernel(self.data, gamma=gamma)
            
            for k in range(2, self.max_clusters + 1):
                try:
                    model = SpectralClustering(
                        n_clusters=k, 
                        affinity='precomputed', 
                        random_state=42, 
                        assign_labels='kmeans'
                    )
                    labels = model.fit_predict(affinity_matrix)
                    
                    if len(set(labels)) > 1:
                        score = silhouette_score(self.data, labels)
                    else:
                        score = -1
                        
                    if score > best_score:
                        best_score = score
                        best_config = (k, gamma)
                except:
                    continue

        self.best_k, self.best_gamma = best_config
        
        if self.best_k:
            print(f"✔ Best Config: k={self.best_k}, gamma={self.best_gamma} | Silhouette Score: {best_score:.4f}")
        else:
            print("⚠ Failed to find optimal config. Using default.")
            self.best_k = 3
            self.best_gamma = 1.0

    def _fit_model(self):
        print(f"Training Spectral Clustering (k={self.best_k})...")
        affinity_matrix = rbf_kernel(self.data, gamma=self.best_gamma)
        self.model = SpectralClustering(
            n_clusters=self.best_k, 
            affinity='precomputed', 
            random_state=42, 
            assign_labels='kmeans'
        )
        self.labels = self.model.fit_predict(affinity_matrix)
        
        # Calculate centroids manually (Spectral doesn't provide them automatically)
        self.centroids = np.zeros((self.best_k, self.data.shape[1]))
        for i in range(self.best_k):
            mask = self.labels == i
            if np.any(mask):
                self.centroids[i] = np.mean(self.data[mask], axis=0)

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        
        # Show Affinity Matrix
        plt.figure(figsize=(6, 5))
        affinity_matrix = rbf_kernel(self.data, gamma=self.best_gamma)
        plt.imshow(affinity_matrix, cmap='viridis')
        plt.colorbar(label='Similarity')
        plt.title(f"Affinity Matrix (gamma={self.best_gamma})")
        plt.show()

        if dims == 2:
            X1, X2 = self.data[:, 0], self.data[:, 1]
            distances = euclidean_distances(self.data, self.centroids)
            radius = [np.max(distances[self.labels == i, i]) if np.any(self.labels==i) else 0 for i in range(len(self.centroids))]

            plt.figure(figsize=(10, 6))
            plt.scatter(X1, X2, c=self.labels, s=40, cmap='viridis', alpha=0.7)
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
            
            for i, centroid in enumerate(self.centroids):
                circle = plt.Circle(centroid, radius[i], color='black', fill=False, alpha=0.3)
                plt.gca().add_patch(circle)
                
            plt.title("Spectral Clustering (2D)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.show()
            
        elif dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels, s=40, cmap='viridis')
            ax.set_title("Spectral Clustering (3D)")
            plt.show()
        else:
             print("Visualization only available for 2D or 3D data.")

    def run_spectral_clustering(self):
        print("=== Running Spectral Clustering ===")
        self._find_optimal_k_gamma()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
