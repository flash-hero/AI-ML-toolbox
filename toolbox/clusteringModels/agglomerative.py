from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from itertools import product

class Agglomerative_Clustering:
    """
    Agglomerative Hierarchical Clustering.
    
    How it works:
    "Bottom-up" approach.
    1. Treats each data point as its own single-point cluster.
    2. Repeatedly merges the two closest clusters together until only 'k' clusters remain.
    
    Why use it?
    - Produces a hierarchy (tree structure) of clusters.
    - Can restrict by 'k' (number of clusters) or distance threshold.
    """

    def __init__(self, data, max_clusters=10):
        self.data = data
        self.max_clusters = max_clusters
        self.best_k = None
        self.best_linkage = None
        self.best_affinity = None
        self.best_score = -1
        self.model = None
        self.labels = None
        # Scikit-learn changed argument name from 'affinity' to 'metric' in version 1.2+
        self.use_metric = self._use_metric_instead_of_affinity()

    def _use_metric_instead_of_affinity(self):
        """
        Checks sklearn version to use correct parameter name.
        """
        version = sklearn.__version__.split('.')
        major, minor = int(version[0]), int(version[1])
        return (major == 1 and minor >= 2) or (major > 1)

    def _find_best_params(self):
        """
        Tests combinations of k, linkage, and affinity to find the best Silhouette Score.
        """
        print("Searching for best hyperparameters...")
        candidate_linkages = ['ward', 'average', 'complete', 'single']
        candidate_affinities = ['euclidean', 'manhattan', 'cosine']
        
        for k in range(2, self.max_clusters + 1):
            for linkage in candidate_linkages:
                # 'ward' linkage only supports 'euclidean' distance
                affinities = ['euclidean'] if linkage == 'ward' else candidate_affinities

                for affinity in affinities:
                    try:
                        if self.use_metric:
                            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=affinity)
                        else:
                            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity=affinity)

                        labels = model.fit_predict(self.data)
                        score = silhouette_score(self.data, labels)

                        if score > self.best_score:
                            self.best_score = score
                            self.best_k = k
                            self.best_linkage = linkage
                            self.best_affinity = affinity
                            
                    except Exception as e:
                        # print(f"Skipping invalid combination: k={k}, linkage={linkage}, affinity={affinity}")
                        continue

        print(f"✔ Best parameters: k={self.best_k}, linkage={self.best_linkage}, "
              f"{'metric' if self.use_metric else 'affinity'}={self.best_affinity}, "
              f"Silhouette Score={self.best_score:.4f}")

    def _fit_model(self):
        """
        Trains the final model.
        """
        if self.use_metric:
            self.model = AgglomerativeClustering(
                n_clusters=self.best_k,
                linkage=self.best_linkage,
                metric=self.best_affinity
            )
        else:
            self.model = AgglomerativeClustering(
                n_clusters=self.best_k,
                linkage=self.best_linkage,
                affinity=self.best_affinity
            )
        self.labels = self.model.fit_predict(self.data)

    def _describe_clusters(self):
        print("\nCluster Description:")
        for label in range(self.best_k):
            count = np.sum(self.labels == label)
            percent = 100 * count / len(self.data)
            print(f"Cluster {label} : {count} points ({percent:.2f}%)")

    def _visualize(self):
        dims = self.data.shape[1]
        data = self.data.values if isinstance(self.data, pd.DataFrame) else self.data
        labels = self.labels

        if dims == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=40)
            plt.title("Agglomerative Clustering (2D)")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.grid(True)
            plt.show()

        elif dims == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10', s=40)
            ax.set_title("Agglomerative Clustering (3D)")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("X3")
            plt.show()
        else:
            print("Visualization only available for 2D or 3D data.")

    def run_clustering_hiérarchique_agglomératif_clustering(self):
        print("=== Running Agglomerative Hierarchical Clustering ===")
        self._find_best_params()
        self._fit_model()
        self._describe_clusters()
        self._visualize()
