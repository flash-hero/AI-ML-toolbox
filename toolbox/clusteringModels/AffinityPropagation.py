import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

class AffinityPropagationClustering:
    """
    Affinity Propagation Clustering.
    
    How it works:
    Unlike K-Means (which needs you to tell it how many clusters 'k' to find), 
    Affinity Propagation finds the number of clusters automatically.
    It works by sending "messages" between data points to find which points are "exemplars" (representatives) of others.
    
    Why use it?
    - You don't need to guess 'k'.
    - Good for finding "representatives" in the data.
    """

    def __init__(self, data):
        self.data = data
        self.best_damping = None
        self.best_preference = None
        self.best_convergence_iter = None
        self.max_iter = None
        self.model = None
        self.labels = None
        self.cluster_centers_indices = None
        self.n_clusters = None

    def _find_optimal_parameters(self, n_iter=20):
        """
        Random Search to find the best settings (hyperparameters).
        """
        print("Searching for optimal parameters using Random Search...")

        best_score = -1
        best_params = {}

        for _ in range(n_iter):
            # Generate random parameters
            damping = np.round(np.random.uniform(0.5, 0.95), 2)
            preference = np.round(np.random.uniform(-300, -10), 1)
            max_iter = int(np.random.choice([200, 500, 1000]))
            convergence_iter = int(np.random.choice([10, 15, 30, 50]))

            try:
                model = AffinityPropagation(
                    damping=damping,
                    preference=preference,
                    max_iter=max_iter,
                    convergence_iter=convergence_iter,
                    random_state=42
                )
                model.fit(self.data)

                # We need at least 2 clusters to calculate Silhouette Score
                if len(np.unique(model.labels_)) > 1:
                    score = silhouette_score(self.data, model.labels_)
                    # print(f"damping={damping}, preference={preference}, score={score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'damping': damping,
                            'preference': preference,
                            'max_iter': max_iter,
                            'convergence_iter': convergence_iter
                        }

            except Exception as e:
                # print(f"Error with parameters: {str(e)}")
                continue

        if best_params:
            self.best_damping = best_params['damping']
            self.best_preference = best_params['preference']
            self.max_iter = best_params['max_iter']
            self.best_convergence_iter = best_params['convergence_iter']
            print(f"\nBest parameters found: {best_params} | Silhouette Score = {best_score:.4f}")
        else:
            self.best_damping = 0.8
            self.best_preference = -50
            self.max_iter = 200
            self.best_convergence_iter = 15
            print("\n⚠ No good score found. Using default values.")

    def _fit_model(self):
        """
        Trains the model with the best parameters found.
        """
        print("\nTraining model with best parameters...")
        self.model = AffinityPropagation(
            damping=self.best_damping,
            preference=self.best_preference,
            max_iter=self.max_iter,
            convergence_iter=self.best_convergence_iter,
            random_state=42
        )
        self.model.fit(self.data)
        self.labels = self.model.labels_
        self.cluster_centers_indices = self.model.cluster_centers_indices_
        self.n_clusters = len(self.cluster_centers_indices)
        print(f"Number of clusters found: {self.n_clusters}")

    def _describe_clusters(self):
        """
        Prints statistics about the found clusters.
        """
        print("\nCluster Description:")
        for i in range(self.n_clusters):
            count = np.sum(self.labels == i)
            percent = 100 * count / len(self.data)
            print(f"Cluster {i} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        """
        Visualizes the clusters in 2D or 3D.
        """
        dims = self.data.shape[1]
        data = self.data
        labels = self.labels

        # Use PCA if more than 3 dimensions
        if dims > 3:
            print("\nReducing dimensions using PCA for visualization...")
            pca = PCA(n_components=3)
            data_reduced = pca.fit_transform(data)
            dims = 3
        else:
            data_reduced = data

        if dims == 1 or dims == 2:
            plt.figure(figsize=(10, 8))
            X1 = data_reduced[:, 0]
            X2 = np.zeros_like(X1) if dims == 1 else data_reduced[:, 1]
            scatter = plt.scatter(X1, X2, c=labels, cmap='viridis', s=50, alpha=0.8)
            
            centers = data_reduced[self.cluster_centers_indices]
            plt.scatter(centers[:, 0], centers[:, 1] if dims > 1 else 0, c='red', marker='X', s=200, label='Centers')
            
            plt.title("Affinity Propagation Clustering (2D)")
            plt.xlabel("Component 1")
            if dims > 1:
                plt.ylabel("Component 2")
            plt.legend()
            plt.colorbar(scatter)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif dims == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2], c=labels, cmap='viridis', s=50)
            
            centers = data_reduced[self.cluster_centers_indices]
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='X', s=200, label='Centers')
            
            ax.set_title("Affinity Propagation Clustering (3D)")
            ax.legend()
            plt.tight_layout()
            plt.show()

    def run_affinity_propagation_clustering(self, n_iter=20):
        print("=== Running Affinity Propagation Clustering ===")
        self._find_optimal_parameters(n_iter=n_iter)
        self._fit_model()
        self._describe_clusters()
        self._visualize()