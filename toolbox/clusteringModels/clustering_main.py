import numpy as np
import pandas as pd

# Import Clustering Models
# Clustering groups similar data points together without knowing the "correct" answer (Unsupervised Learning).
from clusteringModels.kmeans import KMeansClustering
from clusteringModels.kprototypes import KPrototypesClustering
from clusteringModels.kmodes import KModesClustering
from clusteringModels.cmeans import CMeansClustering
from clusteringModels.fuzzy_cmeans import FuzzyCMeansClustering
from clusteringModels.cop_kmeans import COPKMeansClustering
from clusteringModels.kmedoids import KMedoidsClustering
from clusteringModels.clarans import CLARANSClustering
from clusteringModels.DBSCAN import DBSCANClustering
from clusteringModels.hdbscan import HDBSCANClustering
from clusteringModels.meanshift import MeanShiftClustering
from clusteringModels.agglomerative import Agglomerative_Clustering
from clusteringModels.divisivehierarchical import DivisiveHierarchicalClustering
from clusteringModels.birch import BIRCHClustering
from clusteringModels.Agglomerativebirch import AgglomerativeBirchClustering
from clusteringModels.spectral import SpectralClusteringModel
from clusteringModels.minibatchkmeans import MiniBatchKMeansClustering
from clusteringModels.OPTICS import OPTICSClustering
from clusteringModels.AffinityPropagation import AffinityPropagationClustering
from clusteringModels.DensityPeaks import DensityPeaksClustering
from clusteringModels.SelfOrganizingMaps import SOMClustering

# Tool to analyze and visualize the clusters found
from clusteringModels.ClusterAnalyzerVis import ClusterAnalyzer

class ClusteringModels_main:
    """
    This class acts as a Manager (Orchestrator) for all Clustering models.
    It runs the selected algorithms and then analyzes the results.
    """
    
    def __init__(self, data_encoded, data_original):
        """
        Initializes the manager.
        
        Parameters:
        - data_encoded: The numeric data used for the mathematical calculations of clustering.
        - data_original: The original data (with text/categories) used to interpret what the clusters mean.
        """
        # Ensure data is a DataFrame
        if not isinstance(data_encoded, pd.DataFrame):
            data_encoded = pd.DataFrame(data_encoded)

        self.data_encoded = data_encoded
        self.data_original = data_original

        # Identify categorical columns (needed for some specific models like K-Prototypes)
        categorical_columns = [i for i, dtype in enumerate(data_encoded.dtypes) if dtype == 'object']

        # Initialize all available models
        # Note: We pass the data to them immediately
        model_list = [
            ('K-means', KMeansClustering(self.data_encoded)),
            ('K-Prototypes', KPrototypesClustering(self.data_encoded, categorical_columns)),
            ('KModes', KModesClustering(self.data_encoded)),
            ('C-Means', CMeansClustering(self.data_encoded)),
            ('Fuzzy C-means', FuzzyCMeansClustering(self.data_encoded)),
            ('COP-KMeans', COPKMeansClustering(self.data_encoded)),
            ('MiniBatch-KMeans', MiniBatchKMeansClustering(self.data_encoded)),
            ('K-Medoids', KMedoidsClustering(self.data_encoded)),
            ('CLARANS', CLARANSClustering(self.data_encoded)),
            ('DBSCAN', DBSCANClustering(self.data_encoded)),
            ('HDBSCAN', HDBSCANClustering(self.data_encoded)),
            ('Mean-Shift', MeanShiftClustering()),
            ('Agglomerative Hierarchical', Agglomerative_Clustering(self.data_encoded)),
            ('Divisive Hierarchical', DivisiveHierarchicalClustering()),
            ('BIRCH', BIRCHClustering()),
            ('Agglomerative BIRCH', AgglomerativeBirchClustering(self.data_encoded)),
            ('Spectral', SpectralClusteringModel(self.data_encoded)),
            ('OPTICS', OPTICSClustering(self.data_encoded)), 
            ('Affinity Propagation', AffinityPropagationClustering(self.data_encoded)),
            ('Density Peaks', DensityPeaksClustering(self.data_encoded)),
            ('Self-Organizing Maps', SOMClustering(self.data_encoded))
        ]
        
        # Create dictionary of models
        self.models = {i+1: model for i, model in enumerate(model_list)}

    def get_available_models(self):
        """Returns the dictionary of available models."""
        return self.models

    def _execute_model(self, model_name, model_instance):
        """
        Helper method to execute a single model and analyze its results.
        """
        # Construct method name dynamically
        # Example: 'K-means' -> 'run_k_means_clustering'
        method_name = f"run_{model_name.lower().replace('-', '_').replace(' ', '_')}_clustering"

        if hasattr(model_instance, method_name):
            try:
                print("\n" + "-"*80)
                print(f"RUNNING ALGORITHM: {model_name}".center(80))
                print("-"*80)

                # Run the clustering algorithm
                getattr(model_instance, method_name)()

                # --- Post-Clustering Analysis ---
                # After the model creates clusters (assigns labels), we interpret them.
                if hasattr(model_instance, 'labels') and model_instance.labels is not None:
                    # Use the original data to understand the characteristics of each cluster
                    analyzer = ClusterAnalyzer(self.data_original, model_instance.labels)
                    analyzer.run_analysis()
                else:
                    print("Warning: No cluster labels generated.")

                print("\n" + "-"*80)
                print(f"Process '{model_name}' completed successfully.".center(80))
                print("-"*80 + "\n")
                print("="*80 + "\n")

            except Exception as e:
                print(f"\n Error with model '{model_name}': {e}")
                print("Skipping to next model...\n" + "-"*80)
        else:
            print(f"Warning: Method '{method_name}' not found for model {model_name}")

    def run_selected_model(self, selected_indices=None):
        """
        Runs the selected models.
        If selected_indices is None, runs ALL models.
        """
        if selected_indices is None:
             selected_indices = list(self.models.keys())
        
        # Validate indices
        valid_indices = []
        for i in selected_indices:
            try:
                idx = int(i)
                if idx in self.models:
                    valid_indices.append(idx)
            except ValueError:
                continue

        # Run models
        for idx in valid_indices:
            model_name, model_instance = self.models[idx]
            self._execute_model(model_name, model_instance)
