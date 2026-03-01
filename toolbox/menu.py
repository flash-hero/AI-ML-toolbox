import pandas as pd  # Library for data manipulation
import numpy as np   # Library for numerical operations

# Import our custom modules
from vis_analyse import DataAnalyzer
from preprocessor import DataPreprocessor
from data_collection import DataImporter

# Import model orchestrators (managers)
# These handle running the specific machine learning models
from classificationModels.RNNClassificationModels.RNNclassification_main import ClassificationRNNModels_main
from classificationModels.MLClassificationModels.MLclassification_main import ClassificationMLModels_main
from regressionModels.MLRegressionModels.MLregression_main import RegressorMLModels_main
from regressionModels.RNNRegressionModels.RNNregression_main import RegressorRNNModels_main

# Import clustering lazily later to avoid importing heavy/compiled libs at module import time
from feature_selection import FeatureSelector
        
import warnings
warnings.filterwarnings("ignore")  # Hide annoying warning messages to keep the output clean

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU (Force CPU) to avoid complex setup issues

from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_object_dtype

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

class Main:
    """
    The Main class is the control center of the application.
    It handles the User Interface (CLI), loads data, asks the user what to do, 
    and calls the appropriate tools to do it.
    """
    
    def __init__(self):
        """
        Initialize all the tools we might need.
        """
        self.importer = DataImporter()      # Tool to load files
        self.data_analyzer = None           # Tool to visualize data (created later)
        # Initialize interactions with our model categories
        self.mainMLClassifier = ClassificationMLModels_main()
        self.mainMLRegressor = RegressorMLModels_main()
        self.mainRNNClassifier = ClassificationRNNModels_main()
        self.mainRNNRegressor = RegressorRNNModels_main()
        self.mainCluster = None             # Initialized only if needed
        self.data = None                    # Will hold our dataset

    def ask_for_target(self, columns):
        """
        Helper function to ask the user which column they want to predict.
        """
        while True:
            print("\n Available columns:")
            for i, col in enumerate(columns):
                print(f"{i + 1}. {col}")
            
            choice = input("\n Enter the name or number of the Target column (the one you want to predict): ").strip()
            
            # Allow user to enter the number index
            if choice.isdigit() and 1 <= int(choice) <= len(columns):
                return columns[int(choice) - 1]
            # Allow user to enter the name directly
            elif choice in columns:
                return choice
            else:
                print("Invalid choice. Please try again.")

    def handle_column_type_changes(self, preprocessor):
        """
        Interactively asks the user if they want to fix any column data types.
        """
        while True:
            print("\n Do you want to change the data type of any column? (e.g. text to number)")
            print("1. Yes")
            print("2. No")
            choice = input("Enter 1 or 2: ").strip()
            
            if choice == '2':
                break
            elif choice == '1':
                col = self.ask_for_target(preprocessor.data.columns) # Reuse target picker for column picking
                print(f"\n Selected column: {col}")
                print("Choose new type:")
                print("1. int (Integer number)")
                print("2. float (Decimal number)")
                print("3. object (Text)")
                print("4. datetime (Date)")
                type_choice = input("Enter choice (1-4): ").strip()
                
                type_map = {}
                if type_choice == '1': type_map[col] = 'int'
                elif type_choice == '2': type_map[col] = 'float'
                elif type_choice == '3': type_map[col] = 'object'
                elif type_choice == '4': type_map[col] = 'datetime64[ns]'
                
                if type_map:
                    preprocessor.change_dtype(type_map)
                    print(f"Convertion attempted.")
            else:
                print("Invalid choice.")

    def main(self):
        """
        The main loop of the application.
        """
        
        # --- 1. Load Data ---
        # Use our importer tool to select and read a file
        self.data = self.importer.import_data_file()
        if self.data is None:
            print("No data loaded. Exiting... \n")
            return
        print("Data loaded successfully. \n")

        # --- 2. Initial Visualization ---
        # Show basic info (rows, columns, types)
        data_vis = DataAnalyzer(self.data)
        data_vis.info_data()
        
        # --- 3. Preprocessing (Cleaning) ---
        preprocessor = DataPreprocessor(self.data)
        
        # Ask user if they want to correct data types
        self.handle_column_type_changes(preprocessor)
        self.data = preprocessor.data  # Update our main data reference
        
        # Run standard automated cleaning (remove duplicates, handle missing values)
        preprocessed_data = preprocessor.preprocess()
        print("\n Data has been preprocessed (cleaned).")
        print(preprocessed_data.head()) # Show first few rows

        # --- 4. Choose Problem Type ---
        while True:
            print("\n▶ Choose the type of problem you want to solve:")
            print("1. Classification/Regression (Supervised Learning)")
            print("2. Clustering (Unsupervised Learning)")
            choice = input("Enter the number of your choice (1-2): ").strip()

            if choice in {"1", "2"}:
                break
            else:
                print("Invalid input. Please enter '1' or '2'.")

        # --- 5. Classification / Regression Flow ---
        if choice == '1':
            # Ask user which column is the Target (Label)
            target_variable = self.ask_for_target(preprocessor.data.columns) 
            target = preprocessor.data[target_variable] 

            # Create variables for Features (X) and Target (y)
            feature_variable = preprocessor.data.drop(columns=[target_variable])
            
            # --- Feature Selection (Optional) ---
            print("\n▶ Do you want to use statistical methods to select only the most important features?")
            print("1. Yes")
            print("2. No")
            feature_selection_choice = input("Your choice (1-2): ").strip()

            if feature_selection_choice == '1':
                # Use our FeatureSelector tool
                selector = FeatureSelector(feature_variable, target)
                # It returns the reduced X and the y
                X_selected, y_selected = selector.run_feature_selection(feature_variable, target)
                if X_selected is not None:
                    feature_variable = X_selected
                    print("Best features selected successfully.")
                else:
                    print("Feature selection failed or returned no features.")
            else:
                print("Skipping feature selection. Using all columns.")
            
            # Update our working data with only selected features + target
            # (We reconstruct the dataframe to ensure consistency)
            preprocessor.data = pd.concat([feature_variable, target], axis=1)

            # Analyze relationships
            self.data_analyzer = DataAnalyzer(preprocessor.data)
            # (Viz logic here can be expanded)
            
            # --- Normalization and Encoding ---
            # Scale numbers to 0-1 range
            preprocessor.normalize_numeric_columns()
            
            # Convert text categories to numbers (Encoding)
            # We encode the predictors (X)
            preprocessor.encode_categorical_columns(method='onehot')
            
            # We assume target is handled by models or encoded if needed
            # Let's check target type to decide if it's Classification or Regression
            y = preprocessor.data[target_variable]
            
            # Check if it's a Time Series problem
            is_time_series = input(
                "\n▶ Is this a Time Series problem (does order/time matter)?\n"
                "1. Yes\n"
                "2. No\n"
                "Enter your choice (1-2): "
            ).strip()

            if is_time_series == '1':
                 # --- Time Series Flow ---
                datetime_columns = [col for col in self.data.columns if pd.api.types.is_datetime64_any_dtype(self.data[col])]
                
                if datetime_columns:
                    # Sort by date
                    preprocessor.prepare_time_series_data(date_col=datetime_columns[0])
                else:
                    print("No datetime column detected. Proceeding assuming data is already sorted.")
                
                # Split data sequentially (past -> future)
                X_train, X_test, y_train, y_test = preprocessor.split_sequential_data(target_variable)

                # Decide if Classification or Regression based on target type
                # If target is text or few unique integers -> Classification
                if is_object_dtype(y) or is_categorical_dtype(y) or (is_numeric_dtype(y) and y.nunique() < 20):
                    print("Detected Classification problem.")
                    self.mainRNNClassifier.run_models_selected_classifier(X_train, y_train, X_test, y_test)
                else:
                    print("Detected Regression problem.")
                    self.mainRNNRegressor.run_models_selected_regressor(X_train, y_train, X_test, y_test)

            else:
                # --- Standard Flow ---
                print("Standard problem (not time-dependent). Splitting data randomly.")
                X_train, X_test, y_train, y_test = preprocessor.split_data(target_variable)
                
                # Decide if Classification or Regression
                if is_object_dtype(y) or is_categorical_dtype(y) or (is_numeric_dtype(y) and y.nunique() < 20):
                     print("Detected Classification problem.")
                     self.mainMLClassifier.run_models_selected_classifier(X_train, y_train, X_test, y_test)
                elif np.issubdtype(y.dtype, np.number):
                     print("Detected Regression problem.")
                     self.mainMLRegressor.run_models_selected_regressor(X_train, y_train, X_test, y_test)
                else:
                     print("Could not determine problem type from target variable.")

        # --- 6. Clustering Flow ---
        elif choice == '2':
            # Import clustering module lazily
            from clusteringModels.clustering_main import ClusteringModels_main

            self.data_analyzer = DataAnalyzer(preprocessed_data)

            # 1. Normalize numbers
            preprocessor.normalize_numeric_columns()

            # 2. Save original data for analysis (before we mess with it)
            data_original = self.data.copy()

            # 3. Encode text to numbers
            preprocessor.encode_categorical_columns(method='onehot')

            # 4. Select features (Clustering usually uses all, or specific ones)
            # Interactive feature selection for clustering?
            # For now, let's use all preprocessed features
            data_encoded = preprocessor.data

            if data_encoded.shape[1] < 2:
                print("Data must have at least 2 columns for clustering.")
                return

            # 5. Run the Clustering Pipeline
            # We pass encoded data for the math, and original data for interpreting results
            self.mainCluster = ClusteringModels_main(data_encoded, data_original)
            self.mainCluster.run_selected_model()

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    Main().main() 
