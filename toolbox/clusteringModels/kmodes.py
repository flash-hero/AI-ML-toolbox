import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

class KModesClustering:
    """
    K-Modes Clustering.
    
    How it works:
    A variation of K-Means designed specifically for CATEGORICAL data.
    Instead of using Euclidean distance (which makes no sense for categories like "Color=Red"),
    it uses a dissimilarity measure (Hamming distance) and replaces centroids with modes.
    
    Why use it?
    - When your data is textual or categorical (e.g., Blood Type, City, Product Category).
    - Standard K-Means fails on categorical data.
    """

    def __init__(self, data):
        self.data = data
        self.kmode_model = None
        self.clusters = None

    def _find_optimal_k(self):
        """
        Elbow method using Cost to find optimal k.
        """
        print("Finding optimal k using Elbow Method...")
        cost = []
        K = list(range(1, 6))

        for num_clusters in K:
            # Init 'Cao' is a method to initialize centroids
            kmode = KModes(n_clusters=num_clusters, init="Cao", n_init=1, verbose=0)
            kmode.fit_predict(self.data)
            cost.append(kmode.cost_)

        # Calculate relative variation to auto-detect Elbow
        variations = [0]
        for i in range(1, len(cost)):
            delta = (cost[i - 1] - cost[i]) / cost[i - 1] * 100
            variations.append(round(delta, 2))

        # Simple heuristic: max drop
        optimal_k_index = np.argmax(variations[1:]) + 1
        optimal_k = K[optimal_k_index]

        # Results table
        print("\nResults:")
        for k_val, c, v in zip(K, cost, variations):
            print(f"k = {k_val} → Cost = {c:.0f}, Variation = {v:.2f}%")

        # Plot Elbow
        plt.figure(figsize=(8, 5))
        plt.plot(K, cost, marker='o', linestyle='-')
        for i, c in enumerate(cost):
            plt.text(K[i], c, f'{c:.0f}', ha='center', va='bottom')
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Cost')
        plt.title('Elbow Method (K-Modes)')
        plt.grid(True)
        plt.legend()
        plt.show()

        print(f"\n✔ Optimal k detected: {optimal_k}")
        return optimal_k

    def _fit_model(self):
        optimal_k = self._find_optimal_k()
        print(f"Training K-Modes with k={optimal_k}...")
        kmode = KModes(n_clusters=optimal_k, init="Cao", n_init=1, verbose=1)
        self.clusters = kmode.fit_predict(self.data)
        self.kmode_model = kmode

    def _describe_clusters(self):
        if self.clusters is None:
            raise ValueError("Clusters not generated yet.")
        print("\nCluster Description:")
        for label in np.unique(self.clusters):
            count = np.sum(self.clusters == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def run_kmodes_clustering(self):
        print("=== Running K-Modes Clustering ===")
        print("(Note: K-Modes is for categorical data. Ensure your input is categorical/textual)")
        self._fit_model()
        self._describe_clusters()
