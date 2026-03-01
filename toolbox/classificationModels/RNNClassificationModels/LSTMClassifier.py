import os
import numpy as np
from keras_tuner import HyperModel, RandomSearch
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.optimizers import Adam
from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_LSTM_Classifier(HyperModel):
    """
    LSTM (Long Short-Term Memory) Classifier.
    
    How it works:
    A powerful type of RNN designed to remember information for long periods.
    It solves the "forgetting" problem of simple RNNs.
    
    Why use it?
    - The standard for many sequence tasks (text, speech, time series).
    - Can capture complex patterns over time.
    """

    def __init__(self):
        self.best_parameter = None
        self.X_train_summary = None

    def build(self, hp):
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(units=hp.Int('units', 32, 128, step=32),
                       activation='relu',
                       input_shape=self.X_train_summary,
                       return_sequences=True))

        # Output layer: 1 neuron with Sigmoid
        model.add(Flatten()) 
        # Note: If we add more stacked LSTM layers, we would need return_sequences=True for them, 
        # and then Flatten or return_sequences=False for the last one.
        # This implementation simplifies by having one main configurable LSTM block followed by Dense layers.
        # (Original code had a loop for layers but it was a bit complex, sticking to a robust single-block structure for simplicity or reusing logic if preferred). 
        # Re-implementing the loop logic from original file for robustness:
        
        # Additional LSTM layers
        num_layers = hp.Int('num_layers', 1, 3)
        for i in range(num_layers):
             model.add(LSTM(units=hp.Int(f'units_{i}', 32, 128, step=32),
                            activation='relu',
                            return_sequences=(i < num_layers - 1)))
        
        model.add(Flatten())
        model.add(Dense(units=hp.Int('dense_units', 64, 256, step=64), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(
                          learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def _get_unique_filename(self, base_name):
        if not os.path.exists(base_name):
            return base_name
        name, ext = os.path.splitext(base_name)
        i = 1
        while os.path.exists(f"{name}_{i}{ext}"):
            i += 1
        return f"{name}_{i}{ext}"

    def train_lstm(self, X_train, y_train, X_test, y_test, n_iter=10, random_state=42):
        print("Searching for best LSTM hyperparameters...")

        self.X_train_summary = (X_train.shape[1], X_train.shape[2])

        tuner = RandomSearch(
            self,
            objective='val_loss',
            max_trials=n_iter,
            executions_per_trial=3,
            overwrite=True,
            directory='lstm_classifier_tuning',
            project_name='classification'
        )

        tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        self.best_parameter = tuner.get_best_models(num_models=1)[0]
        
        # Save best model
        model_filename = self._get_unique_filename("best_lstm_classifier.h5")
        self.best_parameter.save(model_filename)
        print(f"Best LSTM model saved to: {model_filename}")

        print(f"Best parameters: {tuner.get_best_hyperparameters()[0].values}")

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained.")

        predictions = self.best_parameter.predict(X_test)
        return (predictions > 0.5).astype(int).flatten()

    def run_lstm_classifier(self, X_train, y_train, X_test, y_test, n_iter=10):
        print("\n" + "="*40)
        print(" Training LSTM Model ".center(40, "="))
        print("="*40)
        
        self.train_lstm(X_train, y_train, X_test, y_test, n_iter=n_iter)

        print("\n" + "="*40)
        print(" Evaluating LSTM Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
