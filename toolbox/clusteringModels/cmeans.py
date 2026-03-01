import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz

class CMeansClustering:
    """
    Fuzzy C-Means Clustering.
    
    How it works:
    Similar to K-Means, BUT each data point can belong to *multiple* clusters at the same time 
    with different degrees of membership (probability).
    
    Why use it?
    - Good for overlapping data.
    - Captures uncertainty (e.g., a point might be 70% Cluster A and 30% Cluster B).
    """

    def __init__(self, data, max_clusters=10, max_iter=100, m=2):
        self.data = data
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.m = m # Fuzziness coefficient (usually 2)
        self.centroids = None
        self.labels = None

    def run_c_means_clustering(self):
        print("Training Fuzzy C-Means Model...")
        
        # Verify if data is a DataFrame, if so convert to numpy
        if isinstance(self.data, pd.DataFrame):
            data_values = self.data.values
        else:
            data_values = self.data

        # skfuzzy expects data shape to be (n_features, n_samples)
        data_T = data_values.T 
        
        distortions = []
        silhouette_scores = []
        max_silhouette_score = -1
        max_silhouette_k = -1

        # We try different numbers of clusters (from 2 up to max_clusters)
        for i in range(2, self.max_clusters + 1):
            # cntr = centroids
            # u = fuzzy partition matrix (membership)
            # fpc = fuzzy partition coefficient (0 to 1, higher is better quality)
            cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
                data_T, c=i, m=self.m, error=0.005, maxiter=self.max_iter, init=None
            )
            
            # Hard clustering for evaluation: Assign point to cluster with highest membership
            cluster_labels = np.argmax(u, axis=0)
            
            distortions.append(fpc)
            
            # Use Silhouette Score to evaluate standard clustering quality
            if len(np.unique(cluster_labels)) > 1:
                silhouette_score_val = silhouette_score(data_values, cluster_labels)
            else:
                silhouette_score_val = -1
                
            silhouette_scores.append(silhouette_score_val)
            
            if silhouette_score_val > max_silhouette_score:
                max_silhouette_score = silhouette_score_val
                max_silhouette_k = i

        print(f"Optimal number of clusters (based on Silhouette): {max_silhouette_k}")
        print(f"Best Silhouette Score: {max_silhouette_score:.4f}")

        # Retrain with optimal k
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data_T, c=max_silhouette_k, m=self.m, error=0.005, maxiter=self.max_iter, init=None
        )
        self.centroids = cntr
        self.labels = np.argmax(u, axis=0)

        self._describe_clusters(max_silhouette_k, data_values)
        self._plot_metrics(distortions, silhouette_scores)
        self._visualize(data_values)

    def _describe_clusters(self, k, data):
        print("\nCluster Description:")
        for cluster_label in range(k):
            count = np.sum(self.labels == cluster_label)
            percent = 100 * count / len(data)
            print(f"Cluster {cluster_label}: {count} points ({percent:.2f}%)")

    def _plot_metrics(self, distortions, silhouette_scores):
        # Plot Fuzzy Partition Coefficient (Elbow-like)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, self.max_clusters + 1), distortions, marker='o')
        plt.title('Fuzzy Partition Coefficient (Higher is better)')
        plt.xlabel('k')
        plt.ylabel('FPC')
        plt.grid(True)

        # Plot Silhouette Score
        plt.subplot(1, 2, 2)
        if silhouette_scores:
            plt.plot(range(2, self.max_clusters + 1), silhouette_scores, marker='o', color='orange')
            plt.title('Silhouette Scores')
            plt.xlabel('k')
            plt.ylabel('Score')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def _visualize(self, data):
        dims = data.shape[1]
        
        if dims == 2:
            self.visualisation_2(data, self.centroids, self.labels)
        elif dims == 3:
            self.visualisation_3(data, self.centroids, self.labels)
        else:
            print("Visualization available only for 2D or 3D data.")

    def visualisation_2(self, data, centroids, labels):
        X1 = data[:, 0]
        X2 = data[:, 1]
        
        distances = euclidean_distances(data, centroids)
        radius = [np.max(distances[labels == i, i]) if np.any(labels == i) else 0 for i in range(len(centroids))]

        plt.figure(figsize=(10, 6))
        plt.scatter(X1, X2, c=labels, s=50, cmap='viridis', alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c='red', s=100, label='Centroids')

        for i, centroid in enumerate(centroids):
            circle = plt.Circle(centroid, radius[i], color='black', fill=False, alpha=0.3)
            plt.gca().add_patch(circle)

        plt.title('Fuzzy C-Means Clustering (2D)')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()

    def visualisation_3(self, data, centroids, labels): 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=20)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=100, c='red', label='Centroids', marker='X')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title('Fuzzy C-Means Clustering (3D)')
        ax.legend()
        plt.show()
