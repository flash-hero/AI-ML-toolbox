# Import the RNN Regression model classes
from regressionModels.RNNRegressionModels.GRURegressor import Method_GRU_Regressor
from regressionModels.RNNRegressionModels.LSTMRegressor import  Method_LSTM_Regressor


class RegressorRNNModels_main:
    """
    This class acts as a Manager (Orchestrator) for Recurrent Neural Network (RNN) Regression models.
    It handles models like LSTM and GRU for predicting continuous values.
    """
    
    def __init__(self):
        """
        Initializes the manager.
        """
        model_list = [
            ('LSTM', Method_LSTM_Regressor()),
            ('GRU', Method_GRU_Regressor())        
        ]
        # Create dictionary of models
        self.models = {i + 1: model for i, model in enumerate(model_list)}

    def get_available_models(self):
        """Returns the dictionary of available models."""
        return self.models

    def _execute_model(self, model_name, model_instance, X_train, y_train, X_test, y_test):
        """
        Helper method to execute a single model safely.
        """
        # Construct method name: 'LSTM' -> 'run_lstm_regressor'
        method_name = f"run_{model_name.lower()}_regressor"

        if hasattr(model_instance, method_name):
            print("\n" + "="*80)
            print(f"RUNNING MODEL: {model_name}".center(80))
            print("="*80)
            
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
