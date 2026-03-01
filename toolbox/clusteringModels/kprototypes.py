import numpy as np
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from mpl_toolkits.mplot3d import Axes3D

class KPrototypesClustering:
    """
    K-Prototypes Clustering.
    
    How it works:
    A hybrid of K-Means and K-Modes.
    - K-Means part handles numerical data (Euclidean distance).
    - K-Modes part handles categorical data (Hamming distance).
    
    Why use it?
    - Perfect for MIXED data types (e.g., Age (number) + City (category)).
    """

    def __init__(self, data, categorical_columns, max_clusters=10):
        self.data = data
        self.categorical_columns = categorical_columns  # Index of categorical columns (e.g., [1, 3])
        self.max_clusters = min(max_clusters, len(data))
        self.best_k = None
        self.model = None
        self.labels = None
        self.centroids = None

    def _find_optimal_k(self):
        """
        Elbow Method using Cost.
        """
        print("Finding optimal k for K-Prototypes...")
        best_cost = float('inf')
        costs = []
        valid_ks = []

        for k in range(2, self.max_clusters + 1):
            try:
                # 'Cao' initialization is generally faster
                model = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=0)
                model.fit_predict(self.data, categorical=self.categorical_columns)
                costs.append(model.cost_)
                valid_ks.append(k)

                if model.cost_ < best_cost:
                    best_cost = model.cost_
                    self.best_k = k
            except ValueError as e:
                # Sometimes KPrototypes fails with small k or specific data init
                continue

        if self.best_k is None:
             # Fallback
             self.best_k = 3

        self._plot_cost(valid_ks, costs)
        print(f"✔ Best k (lowest cost): {self.best_k} | Cost: {best_cost:.4f}")
        print("(Note: For Elbow method, you usually pick the 'bend', not strictly the lowest cost. Check the plot!)")

    def _fit_model(self):
        print(f"Training K-Prototypes with k={self.best_k}...")
        self.model = KPrototypes(n_clusters=self.best_k, init='Cao', n_init=5, verbose=0)
        self.labels = self.model.fit_predict(self.data, categorical=self.categorical_columns)
        self.centroids = self.model.cluster_centroids_

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _plot_cost(self, ks, costs):
        plt.figure(figsize=(8, 5))
        plt.plot(ks, costs, marker='o')
        plt.title("Elbow Method (K-Prototypes Cost)")
        plt.xlabel("k")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.show()

    def run_k_prototypes_clustering(self):
        print("=== Running K-Prototypes Clustering ===")
        self._find_optimal_k()
        self._fit_model()
        self._describe_clusters()
