import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

class FuzzyCMeansClustering:
    """
    Fuzzy C-Means (Manual Implementation).
    
    How it works:
    This is a "from scratch" implementation of the Fuzzy C-Means algorithm.
    It allows multiple memberships (a point can belong to Cluster A with 0.8 probability and Cluster B with 0.2).
    
    Why use it?
    - Educational: Good to understand how the math works under the hood.
    - Soft clustering: Handling ambiguity in data.
    """

    def __init__(self, data, k_range=range(2, 11), m_values=np.arange(1.5, 3.1, 0.1), max_iter=150, error=1e-5):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.k_range = k_range
        self.m_values = m_values
        self.max_iter = max_iter
        self.error = error
        self.best_k = None
        self.best_m = None
        self.U = None
        self.centers = None
        self.labels = None

    def _init_membership(self, n, k):
        """Randomly initialize membership matrix."""
        U = np.random.rand(n, k)
        return U / np.sum(U, axis=1, keepdims=True)

    def _update_centers(self, U, m, data):
        """Update cluster centers based on current memberships."""
        um = U ** m
        return (um.T @ data) / np.sum(um.T, axis=1, keepdims=True)

    def _update_membership(self, data, centers, m):
        """Update membership matrix based on distance to new centers."""
        dist = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        dist = np.fmax(dist, np.finfo(np.float64).eps) # Avoid division by zero
        power = 2 / (m - 1)
        temp = dist[:, :, np.newaxis] / dist[:, np.newaxis, :]
        return 1.0 / np.sum(temp ** power, axis=2)

    def _fit_model(self, k, m):
        """Trains a single model with specific k and m."""
        n, d = self.data.shape
        U = self._init_membership(n, k)
        for _ in range(self.max_iter):
            centers = self._update_centers(U, m, self.data)
            U_new = self._update_membership(self.data, centers, m)
            if np.linalg.norm(U_new - U) < self.error:
                break
            U = U_new
        labels = np.argmax(U, axis=1)
        return U, centers, labels

    def _find_optimal_k_and_m(self):
        """
        Grid Search to find best 'k' (num clusters) and 'm' (fuzziness coefficient).
        """
        print("Optimizing k and m...")
        best_score = -1
        best_k = None
        best_m = None
        best_U, best_centers, best_labels = None, None, None
        scores = []

        for m in self.m_values:
            for k in self.k_range:
                try:
                    U, centers, labels = self._fit_model(k, m)
                    
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(self.data, labels)
                    else:
                        score = -1
                        
                    scores.append((k, m, score))

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_m = m
                        best_U = U
                        best_centers = centers
                        best_labels = labels

                except Exception as e:
                    # print(f"Error for k={k}, m={m:.2f} : {e}")
                    continue

        self.best_k = best_k
        self.best_m = best_m
        self.U = best_U
        self.centers = best_centers
        self.labels = best_labels

        self._plot_heatmap(scores)
        print(f"\n✔ Best Fit: k={best_k}, m={best_m:.2f} | Silhouette Score: {best_score:.4f}")

    def _plot_heatmap(self, scores):
        import seaborn as sns
        
        df = pd.DataFrame(scores, columns=["k", "m", "score"])
        df["m"] = df["m"].round(3)       
        pivot = df.pivot(index="m", columns="k", values="score")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Silhouette Score (k vs m)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Fuzziness (m)")
        plt.show()

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        data = self.data
        labels = self.labels
        centers = self.centers

        if dims == 2:
            X1, X2 = data[:, 0], data[:, 1]
            distances = euclidean_distances(data, centers)
            radius = [np.max(distances[labels == i, i]) if np.any(labels==i) else 0 for i in range(len(centers))]

            plt.figure(figsize=(10, 8))
            plt.scatter(X1, X2, c=labels, s=40, cmap='viridis', alpha=0.6)
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centroids')

            for i, center in enumerate(centers):
                circle = plt.Circle(center, radius[i], color='black', fill=False, alpha=0.3)
                plt.gca().add_patch(circle)

            plt.title("Fuzzy C-Means (2D - Manual Impl)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.show()

        elif dims == 3:
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=40)
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=100, label='Centroids')
            ax.set_title("Fuzzy C-Means (3D - Manual Impl)")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("X3")
            plt.legend()
            plt.show()
        else:
            print("Visualization only available for 2D or 3D data.")

    def run_fuzzy_c_means_clustering(self):
        print("=== Running Fuzzy C-Means (Manual) Clustering ===")
        self._find_optimal_k_and_m()
        self._describe_clusters()
        self._visualize()
