# Import all the specific Machine Learning model classes
# Each of these classes contains the logic to train and test one specific algorithm.
from classificationModels.MLClassificationModels.KNNClassifier import Method_KNN_Classifier
from classificationModels.MLClassificationModels.DecisionTreeClassifier import Method_DecisionTree_Classifier
from classificationModels.MLClassificationModels.RandomForestClassifier import Method_RandomForest
from classificationModels.MLClassificationModels.LDAClassifier import Method_LDA_Classifier
from classificationModels.MLClassificationModels.NaiveBayesClassifier import Method_NaiveBayes_Classifier
from classificationModels.MLClassificationModels.LogisticRegressionClassifier import Method_LogisticRegression_Classifier
from classificationModels.MLClassificationModels.SVMClassifier import Method_SVM_Classifier
from classificationModels.MLClassificationModels.QDAClassifier import Method_QDA_Classifier
from classificationModels.MLClassificationModels.XGBoostClassifier import Method_XGBoost_Classifier
from classificationModels.MLClassificationModels.LightGBMClassifier import Method_LightGBM_Classifier
from classificationModels.MLClassificationModels.CatBoostClassifier import Method_CatBoost_Classifier
from classificationModels.MLClassificationModels.AdaBoostClassifier import Method_AdaBoost_Classifier
from classificationModels.MLClassificationModels.BaggingClassifier import Method_Bagging_Classifier
from classificationModels.MLClassificationModels.ExtraTreesClassifier import Method_ExtraTrees_Classifier

class ClassificationMLModels_main:
    """
    This class acts as a Manager (Orchestrator) for all Machine Learning Classification models.
    It stores a list of available models and allows running them one by one or all together.
    """

    def __init__(self):
        """
        Initializes the manager by creating instances of all model classes.
        We store them in a dictionary for easy access.
        """
        model_list = [
            ('Logistic Regression', Method_LogisticRegression_Classifier()),
            ('KNN', Method_KNN_Classifier()),
            ('Decision Tree', Method_DecisionTree_Classifier()),
            ('Random Forest', Method_RandomForest()),
            ('Naive Bayes', Method_NaiveBayes_Classifier()),
            ('SVM', Method_SVM_Classifier()),
            ('LDA', Method_LDA_Classifier()),
            ('QDA', Method_QDA_Classifier()),
            ('XGBoost', Method_XGBoost_Classifier()),
            ('LightGBM', Method_LightGBM_Classifier()),
            ('CatBoost', Method_CatBoost_Classifier()),
            ('AdaBoost', Method_AdaBoost_Classifier()),
            ('Bagging', Method_Bagging_Classifier()),
            ('Extra Trees', Method_ExtraTrees_Classifier())
        ]
        # Create a dictionary where keys are 1, 2, 3... and values are the models
        self.models = {i+1: idx_model for i, idx_model in enumerate(model_list)}


    def get_available_models(self):
        """
        Returns the dictionary of all available models.
        Useful for listing them to the user.
        """
        return self.models

    def _execute_model(self, model_name, model_instance, X_train, y_train, X_test, y_test):
        """
        Helper method to execute a single model safely.
        """
        # We construct the method name dynamically.
        # Example: If model name is 'Random Forest', we look for 'run_random_forest_classifier'
        method_name = f"run_{model_name.lower().replace(' ', '_')}_classifier"

        if hasattr(model_instance, method_name):
            print("\n" + "-"*80)
            print(f"RUNNING MODEL: {model_name}".center(80))
            print("-"*80)
            
            # Call the method dynamically
            getattr(model_instance, method_name)(X_train, y_train, X_test, y_test)
            
            print("\n" + "-"*80)
            print(f"Process '{model_name}' completed successfully.".center(80))
            print("-"*80 + "\n")
        else:
            print(f"Error: Method '{method_name}' not found for model {model_name}")

    def run_models_selected_classifier(self, X_train, y_train, X_test, y_test, selected_indices=None):
        """
        Runs the selected models on the provided training and testing data.
        
        Parameters:
        - X_train, y_train: Training data and labels.
        - X_test, y_test: Testing data and labels.
        - selected_indices: A list of numbers (e.g. [1, 3]) representing which models to run.
                            If None, runs ALL models.
        """
        # If no specific models are selected, default to ALL models
        if selected_indices is None:
            selected_indices = list(self.models.keys())
        
        # Ensure we are working with a list of integers
        # (Handles cases where input might be strings like ['1', '2'])
        valid_indices = []
        for i in selected_indices:
            try:
                idx = int(i)
                if idx in self.models:
                    valid_indices.append(idx)
            except ValueError:
                continue
        
        # Loop through each selected model and run it
        for idx in valid_indices:
            model_name, model_instance = self.models[idx]
            self._execute_model(model_name, model_instance, X_train, y_train, X_test, y_test)