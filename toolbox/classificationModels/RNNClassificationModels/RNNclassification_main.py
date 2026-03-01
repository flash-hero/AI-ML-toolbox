# Import the RNN model classes
# These are deep learning models designed for sequence data (like time series or text).
from classificationModels.RNNClassificationModels.GRUClassifier import Method_GRU_Classifier 
from classificationModels.RNNClassificationModels.LSTMClassifier import Method_LSTM_Classifier


class ClassificationRNNModels_main:
    """
    This class acts as a Manager (Orchestrator) for Recurrent Neural Network (RNN) Classification models.
    It handles models like LSTM and GRU.
    """

    def __init__(self):
        """
        Initializes the manager.
        """
        model_list = [
            ('LSTM', Method_LSTM_Classifier()), 
            ('GRU', Method_GRU_Classifier())
        ]
        # Create a dictionary of models, starting index at 1
        self.models = {i+1: model for i, model in enumerate(model_list)}

    def get_available_models(self):
        """Returns the dictionary of available models."""
        return self.models

    def _execute_model(self, model_name, model_instance, X_train, y_train, X_test, y_test):
        """
        Helper method to execute a single model safely.
        """
        # Construct method name dynamically: 'LSTM' -> 'run_lstm_classifier'
        method_name = f"run_{model_name.lower().replace(' ', '_')}_classifier"

        if hasattr(model_instance, method_name):
            print("\n" + "-"*80)
            print(f"RUNNING MODEL: {model_name}".center(80))
            print("-"*80)
            
            getattr(model_instance, method_name)(X_train, y_train, X_test, y_test)
            
            print("\n" + "-"*80)
            print(f"Process '{model_name}' completed successfully.".center(80))
            print("-"*80 + "\n")
        else:
            print(f"Error: Method '{method_name}' not found for model {model_name}")

    def run_models_selected_classifier(self, X_train, y_train, X_test, y_test, selected_indices=None):
        """
        Runs the selected models.
        If selected_indices is None, runs all available models.
        """
        if selected_indices is None:
            selected_indices = list(self.models.keys())
            
        # Ensure indices are valid integers
        valid_indices = []
        for i in selected_indices:
            try:
                idx = int(i)
                if idx in self.models:
                    valid_indices.append(idx)
            except ValueError:
                continue

        # Run each selected model
        for idx in valid_indices:
            model_name, model = self.models[idx]
            self._execute_model(model_name, model, X_train, y_train, X_test, y_test)
