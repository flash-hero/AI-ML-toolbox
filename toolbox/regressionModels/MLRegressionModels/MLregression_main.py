# Import the Machine Learning Regression models
from regressionModels.MLRegressionModels.KNNRegressor import Method_KNN_Regressor
from regressionModels.MLRegressionModels.DecisionTreeRegressor import Method_DecisionTree
from regressionModels.MLRegressionModels.RandomForestRegressor import Method_RandomForest
from regressionModels.MLRegressionModels.LinearRegressor import Method_Linear_Regressor
from regressionModels.MLRegressionModels.BayesianLinearRegressor import Method_BayesianLinear_Regressor
from regressionModels.MLRegressionModels.BayesianRidgeRegressor import Method_BayesianRidge_Regressor
from regressionModels.MLRegressionModels.GaussianProcessRegressor import Method_GaussianProcess_Regressor
from regressionModels.MLRegressionModels.GradientBoostingRegressor import Method_GradientBoosting_Regressor
from regressionModels.MLRegressionModels.AdaBoostRegressor import Method_AdaBoost_Regressor
from regressionModels.MLRegressionModels.SupportVectorRegressor import Method_SVR_Regressor
from regressionModels.MLRegressionModels.XGBoostRegressor import Method_XGBoost_Regressor
from regressionModels.MLRegressionModels.CatBoostRegressor import Method_Catboost_Regressor
from regressionModels.MLRegressionModels.ExtraTreesRegressor import Method_ExtraTrees_Regressor
from regressionModels.MLRegressionModels.LightGBMRegressor import Method_LightGBM_Regressor

class RegressorMLModels_main:
    """
    This class acts as a Manager (Orchestrator) for all Machine Learning Regression models.
    Regression models predict a continuous number (e.g., Price, Temperature, Sales).
    """
    
    def __init__(self):
        """
        Initializes the manager by creating instances of all model classes.
        """
        model_list = [
            ('KNN', Method_KNN_Regressor()),
            ('Decision Tree', Method_DecisionTree()),
            ('Random Forest', Method_RandomForest()),
            ('SVR', Method_SVR_Regressor()),
            ('Linear regression', Method_Linear_Regressor()),
            ('Bayesian linear', Method_BayesianLinear_Regressor()),
            ('Bayesian Ridge', Method_BayesianRidge_Regressor()),
            ('Gaussian Process regression', Method_GaussianProcess_Regressor()),
            ('Gradient Boosting', Method_GradientBoosting_Regressor()),
            ('AdaBoost', Method_AdaBoost_Regressor()),
            ('XGBoost',Method_XGBoost_Regressor()),
            ('Catboost', Method_Catboost_Regressor()),
            ('Extra trees', Method_ExtraTrees_Regressor()),
            ('LightGBM', Method_LightGBM_Regressor())
        ]
        # Create a dictionary where keys are 1, 2, 3... and values are the models
        self.models = {i + 1: model for i, model in enumerate(model_list)}

    def get_available_models(self):
        """
        Returns the dictionary of all available models.
        """
        return self.models

    def _execute_model(self, model_name, model_instance, X_train, y_train, X_test, y_test):
        """
        Helper method to execute a single model safely.
        """
        # Construct the method name dynamically.
        # Example: 'KNN' -> 'run_knn_regressor'
        method_name = f"run_{model_name.lower().replace(' ', '_')}_regressor"

        if hasattr(model_instance, method_name):
            print("\n" + "="*80)
            print(f"RUNNING MODEL: {model_name}".center(80))
            print("="*80)
            
            # Call the method
            getattr(model_instance, method_name)(X_train, y_train, X_test, y_test)
            
            print("\n" + "-"*80)
            print(f"Process '{model_name}' completed successfully.".center(80))
            print("-"*80 + "\n")
        else:
            print(f"Error: Method '{method_name}' not found for model {model_name}")

    def run_models_selected_regressor(self, X_train, y_train, X_test, y_test, selected_indices=None):
        """
        Runs the selected models.
        
        Parameters:
        - X_train, y_train: Training data and labels.
        - X_test, y_test: Testing data and labels.
        - selected_indices: A list of indices of the models to run. 
                            If None, ALL models will be executed.
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
            self._execute_model(model_name, model_instance, X_train, y_train, X_test, y_test)
