import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall

class CLARANSClustering:
    """
    CLARANS (Clustering Large Applications based on RANdomized Search).
    
    How it works:
    An extension of K-Medoids.
    Instead of checking every single possible swap of medoids (like standard PAM algorithm which is slow),
    it checks only a random sample of neighbors.
    
    Why use it?
    - More efficient than PAM/K-Medoids for larger datasets.
    - More robust to outliers than K-Means (because it uses medoids - actual points - instead of centroids - averages).
    """

    def __init__(self, data, max_clusters=10):
        # Ensure data is numpy array
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = np.array(data)
            
        self.max_clusters = max_clusters
        self.best_k = None
        self.labels = None
        self.clusters = None
        self.medoids = None

    def _find_optimal_k(self):
        """
        Tests multiple k values to find the best one based on Silhouette Score.
        """
        print("Finding optimal k for CLARANS...")
        silhouette_scores = []
        best_score = -1
        
        # numlocal: Number of local minima obtained (iterations)
        # maxneighbor: Max number of neighbors examined
        numlocal = 1 
        maxneighbor = 5

        for k in range(2, self.max_clusters + 1):
            # Run CLARANS
            instance = clarans(self.data.tolist(), k, numlocal, maxneighbor)
            _, _ = timedcall(instance.process)
            
            clusters = instance.get_clusters()
            
            # Convert cluster list-of-lists to labels array
            labels = [None] * len(self.data)
            for cluster_idx, indices in enumerate(clusters):
                for i in indices:
                    labels[i] = cluster_idx
            
            # Remove any potentially unassigned points (shouldn't happen with valid clarans execution but safety check)
            # labels should be fully populated
            
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)
            
            if score > best_score:
                best_score = score
                self.best_k = k
                
        self._plot_silhouette(silhouette_scores)
        print(f"Best k: {self.best_k} | Silhouette Score: {best_score:.4f}")

    def _fit_model(self):
        """
        Train the model with the best k.
        """
        print(f"Training CLARANS with k={self.best_k}...")
        # numlocal=1, maxneighbor=5 (parameters influencing speed/accuracy trade-off)
        instance = clarans(self.data.tolist(), self.best_k, 1, 5)
        _, _ = timedcall(instance.process)
        
        self.clusters = instance.get_clusters()
        self.medoids = instance.get_medoids() # Returns indices of medoids
        
        # Convert to label array
        self.labels = [0] * len(self.data)
        for cluster_idx, indices in enumerate(self.clusters):
            for i in indices:
                self.labels[i] = cluster_idx

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(np.array(self.labels) == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        # Use PCA to project to 2D if dimensions > 2
        if self.data.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.data)
            # Medoids need to be transformed too
            medoid_coords = pca.transform([self.data[m, :] for m in self.medoids])
        else:
            X_pca = self.data
            medoid_coords = [self.data[m, :] for m in self.medoids]
            medoid_coords = np.array(medoid_coords)

        plt.figure(figsize=(8, 6))
        for i in range(self.best_k):
            points = X_pca[np.array(self.labels) == i]
            plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}", alpha=0.5)
            
        plt.scatter(medoid_coords[:, 0], medoid_coords[:, 1], c='black', marker='X', s=200, label='Medoids')
        plt.title("CLARANS Clustering (PCA Projected)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_silhouette(self, scores):
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, self.max_clusters + 1), scores, marker='o')
        plt.title("Silhouette Scores")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    def run_clarans_clustering(self):
        print("=== Running CLARANS Clustering ===")
        self._find_optimal_k()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
