import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from mpl_toolkits.mplot3d import Axes3D

class SOMClustering:
    """
    Self-Organizing Maps (SOM).
    
    How it works:
    A type of Neural Network used for clustering and dimensionality reduction.
    It maps high-dimensional data onto a 2D grid of "neurons".
    Points that end up in the same neuron (or nearby neurons) are similar.
    
    Why use it?
    - Good for visualizing high-dimensional data structure in 2D.
    - Preserves topological properties (neighbors in input space stay neighbors in SOM).
    """

    def __init__(self, data, iterations=1000):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.iterations = iterations
        self.best_params = None
        self.best_score = -1
        self.som = None
        self.labels = None
        self.centroids = None

    def _generate_integer_labels(self, som):
        """Converts SOM (x,y) coordinates into a single integer label."""
        # Label = x * width + y
        width = som._weights.shape[1]
        labels = []
        for x_point in self.data:
            winner = som.winner(x_point)
            labels.append(winner[0] * width + winner[1])
        return np.array(labels)

    def _optimize_hyperparameters(self):
        """
        Grid Search for best Grid Dimensions, Sigma, and Learning Rate.
        """
        print("Optimizing SOM parameters...")
        param_grid = ParameterGrid({
            'x_dim': [5, 10],           # Width of the map
            'y_dim': [5, 10],           # Height of the map
            'sigma': [0.5, 1.0, 2.0],   # Neighborhood radius
            'learning_rate': [0.3, 0.5, 0.8]
        })

        for params in param_grid:
            try:
                som = MiniSom(
                    x=params['x_dim'],
                    y=params['y_dim'],
                    input_len=self.data.shape[1],
                    sigma=params['sigma'],
                    learning_rate=params['learning_rate']
                )
                som.train(self.data, self.iterations)

                labels = self._generate_integer_labels(som)
                
                # Verify we have meaningful clusters
                if len(set(labels)) <= 1 or len(set(labels)) == len(self.data):
                    continue

                score = silhouette_score(self.data, labels)
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self.som = som
                    self.labels = labels
            except Exception as e:
                continue

        if self.best_params:
            print(f"✔ Best Config: {self.best_params}")
            print(f"✔ Best Silhouette Score: {self.best_score:.4f}")
        else:
            print("⚠ No valid clustering found. Try increasing iterations or changing grid size.")

    def _fit_model(self):
        if self.som:
             # Already trained in optimization loop
             self.centroids = np.array([
                self.som.get_weights()[x, y]
                for x, y in [self.som.winner(d) for d in self.data]
            ])
             return

    def _describe_clusters(self):
        if self.labels is None: return
        print("\nCluster Description:")
        unique_labels = sorted(set(self.labels))
        for label in unique_labels:
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster/Cell {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        if self.labels is None: return

        if dims == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='tab20', s=40)
            plt.title("SOM Clustering (2D)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.show()
        elif dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels, cmap='tab20', s=40)
            ax.set_title("SOM Clustering (3D)")
            plt.show()
        else:
            print("Visualization only available for 2D or 3D data.")

    def run_self_organizing_maps_clustering(self):
        print("=== Running SOM Clustering ===")
        self._optimize_hyperparameters()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
