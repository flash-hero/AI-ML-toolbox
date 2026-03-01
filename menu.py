import pandas as pd
import numpy as np

from vis_analyse import DataAnalyzer
from preprocessor import DataPreprocessor
from data_collection import DataImporter

from classificationModels.RNNClassificationModels.RNNclassification_main import ClassificationRNNModels_main
from classificationModels.MLClassificationModels.MLclassification_main import ClassificationMLModels_main

from regressionModels.MLRegressionModels.MLregression_main import RegressorMLModels_main
from regressionModels.RNNRegressionModels.RNNregression_main import RegressorRNNModels_main

# Import clustering lazily later to avoid importing heavy/compiled libs at module import time
# (some optional dependencies like sklearn_extra may be missing or incompatible on the host)
from feature_selection import FeatureSelector
        
import warnings
warnings.filterwarnings("ignore")

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_object_dtype

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

class Main:
    def __init__(self):
        self.importer = DataImporter()
        self.data_analyzer = None
        self.mainMLClassifier = ClassificationMLModels_main()
        self.mainMLRegressor = RegressorMLModels_main()
        self.mainRNNClassifier = ClassificationRNNModels_main()
        self.mainRNNRegressor = RegressorRNNModels_main()
        self.mainCluster = None
        self.data = None

    def main(self):
        # Chargement des données
        # self.data = self.importer.import_data_file()
        # if self.data is None:
        #     print("Aucune donnée chargée. Sortie... \n")
        #     return
        # print("Données chargées avec succès. \n")

        # # Visualisation initiale des données
        # data_vis = DataAnalyzer(self.data)
        # print(data_vis.visualization())

        # Chargement des données
        self.data = self.importer.import_data_file()
        if self.data is None:
            print("Aucune donnée chargée. Sortie... \n")
            return
        print("Données chargées avec succès. \n")

        data_vis = DataAnalyzer(self.data)
        data_vis.info_data()
        
        preprocessor = DataPreprocessor(self.data)
        preprocessor.ask_change_dtype()
        self.data = preprocessor.data  # Mettre à jour les données avec les types modifiés ou non



        # Visualisation avec types corrects
        print(data_vis.visualization())
        
        # Préparation des données
        if isinstance(self.data, pd.DataFrame):
            # Appeler la fonction pour définir l'index
            preprocessor = DataPreprocessor(self.data)
            preprocessed_data = preprocessor.preprocess()
            print("Voici les données prétraitées :")
            print(preprocessed_data)

        while True:  # Modifiée par l'équipe du test
            print("\n▶ Choisissez le type de problème :")
            print("1. Classification/Régression")
            print("2. Clustering")
            choice = input("Entrez le numéro correspondant à votre choix (1-2) : ").strip()

            if choice in {"1", "2"}:
                break
            else:
                print("Entrée invalide. Veuillez entrer '1' pour Classification/Régression ou '2' pour Clustering.")

        if choice == '1':
            # Choix de la variable cible y
            target_variable  = preprocessor.determine_target() 
            target = preprocessor.data[target_variable ] 

            # Les variables X  
            self.data_analyzer = DataAnalyzer(preprocessed_data)
            feature_variable = preprocessor.data.drop(columns=target_variable)
            # Ajouter la sélection de caractéristiques supervisée
            print("\n▶ Souhaitez-vous appliquer une méthode statistique pour sélectionner les variables les plus importantes de votre dataset?")
            print("1. Oui")
            print("2. Non")
            feature_selection_choice = input("Votre choix (1-2) : ").strip()

            if feature_selection_choice == '1':
                selector = FeatureSelector(feature_variable, target)
                result = selector.run_feature_selection(feature_variable, target)
                if result is None:
                    print("La sélection des variables statistiquement les plus importantes de votre dataset a échoué ou a été annulée.")
                    return
                else:
                    feature_variable, target = result
                    print("La sélection des variables statistiquement les plus importantes de votre dataset a été effectuée avec succès.")
            else:
                print("L'étape de la sélection des variables statistiquement les plus importantes de votre dataset a été sautée.")
            
            # À ce stade, `preprocessed_data` et `target_variable` sont définis
            self.data_analyzer = DataAnalyzer(preprocessed_data)
            # print("Exploration des relations entre les caractéristiques et la cible.")
            
            # Définir `target` après avoir obtenu `target_variable`
            y = preprocessor.data[target_variable]

            # Maintenant on passe à la Normalisation et encodage
            preprocessor.normalize_numeric_columns()
            
            new_target_variable_= preprocessor.encode_categorical_columns(target_variable)
            target = preprocessor.data[target_variable]
            
            # Vérifier la présence de colonnes datetime
            is_time_series = input(
            "\n▶ Le problème à résoudre est-il de nature chronologique ?\n"
            "1. Oui\n"
            "2. Non\n"
            "Veuillez entrer votre choix (1-2) : "
            ).strip()
            if is_time_series == '1':
                datetime_columns = [
                    col for col in self.data.columns
                    if pd.api.types.is_datetime64_any_dtype(self.data[col])
                ]
                if datetime_columns:
                    preprocessor.prepare_time_series_data()
                else:
                    print("Aucune colonne de type datetime détectée.")
                X_train, X_test, y_train, y_test = preprocessor.split_sequential_data(preprocessor.select_features_to_keep(target_variable), target)

                
                if target_variable == new_target_variable_ :

                    if isinstance(y.values[0], object) or isinstance(y.values[0], str) or pd.api.types.is_categorical_dtype(y):
                        # Appel des modèles de classification
                        self.mainRNNClassifier.run_models_selected_classifier(X_train, y_train, X_test, y_test)
                    elif np.issubdtype(y.dtype, np.number):
                        # Appel des modèles de régression
                        self.mainRNNRegressor.run_models_selected_regressor(X_train, y_train, X_test, y_test)
                else:
                    print("Aucune colonne de type datetime détectée.")
                X_train, X_test, y_train, y_test = preprocessor.split_sequential_data(preprocessor.select_features_to_keep(target_variable), target)
            else:
                print("Le problème n'est pas de nature chronologique. Méthode ignorée.")
                X_train, X_test, y_train, y_test = preprocessor.split_data(preprocessor.select_features_to_keep(target_variable), target)
                
                if target_variable == new_target_variable_ :
                    print(y.dtype)
                    print(y.values)
                    if is_object_dtype(y) or is_categorical_dtype(y) or isinstance(y.iloc[0], str):
                        # Appel des modèles de classification
                        self.mainMLClassifier.run_models_selected_classifier(X_train, y_train, X_test, y_test)
                    elif np.issubdtype(y.dtype, np.number):
                        # Appel des modèles de régression
                        self.mainMLRegressor.run_models_selected_regressor(X_train, y_train, X_test, y_test)
                else :
                    print("ne sont pas les meme")


        elif choice == '2':
            # Import clustering module here to avoid importing heavy compiled dependencies
            # at top-level (which can fail on some systems / envs).
            from clusteringModels.clustering_main import ClusteringModels_main

            self.data_analyzer = DataAnalyzer(preprocessed_data)

            # 1. Normalisation
            preprocessor.normalize_numeric_columns()

            # 2. Sauvegarder les données originales AVANT encodage
            data_original = self.data.copy()

            # 3. Encodage
            self.data = preprocessor.encode_categorical_columns()

            # 4. Sélection des variables à utiliser pour le clustering
            data_encoded = preprocessor.select_features_to_keep_for_clustering()

            if data_encoded.shape[1] < 2:
                print("Les données doivent contenir au moins 2 caractéristiques pour le clustering.")
                return

            # 5. Lancer le pipeline avec les deux jeux : encodé pour le clustering, original pour l’analyse
            self.mainCluster = ClusteringModels_main(data_encoded, data_original)

            # self.mainCluster = ClusteringModels_main(self.data, data_original)  # data encodé, self.data = original
            self.mainCluster.run_selected_model()

        else:
            print("Choix non valide.")

if __name__ == "__main__":
    Main().main() 
