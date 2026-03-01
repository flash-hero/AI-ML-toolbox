import os
import numpy as np
from keras_tuner import HyperModel, RandomSearch
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.optimizers import Adam
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_LSTM_Regressor(HyperModel):
    """
    LSTM Regressor.
    
    How it works:
    Long Short-Term Memory Network for Regression.
    Capabale of capturing long-term dependencies in the data sequence.
    
    Why use it?
    - Standard choice for sequence prediction problems.
    """

    def __init__(self):
        self.best_parameter = None
        self.X_train_summary = None

    def build(self, hp):
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(units=hp.Int('units_first', 32, 128, step=32),
                       activation='relu',
                       input_shape=self.X_train_summary,
                       return_sequences=True))

        # Additional LSTM layers
        num_layers = hp.Int('num_layers', 1, 3)
        for i in range(num_layers):
            return_seq = i < num_layers - 1
            model.add(LSTM(units=hp.Int(f'units_{i}', 32, 128, step=32),
                           activation='relu',
                           return_sequences=return_seq))

        model.add(Flatten())
        model.add(Dense(units=hp.Int('dense_units', 64, 256, step=64), activation='relu'))
        
        # Output layer: 1 neuron with Linear activation
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer=Adam(
                          learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                      loss='mean_squared_error',
                      metrics=['mae'])
        return model

    def _get_unique_filename(self, base_name):
        if not os.path.exists(base_name):
            return base_name
        name, ext = os.path.splitext(base_name)
        i = 1
        while os.path.exists(f"{name}_{i}{ext}"):
            i += 1
        return f"{name}_{i}{ext}"

    def train_lstm(self, X_train, y_train, X_test, y_test, n_iter=10, epochs=10):
        self.X_train_summary = (X_train.shape[1], X_train.shape[2])

        tuner = RandomSearch(
            self,
            objective='val_loss',
            max_trials=n_iter,
            executions_per_trial=2,
            overwrite=True,
            directory='lstm_tuning',
            project_name='lstm_regression'
        )

        print("Searching for best LSTM hyperparameters...")
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
        tuner.results_summary()

        best_model = tuner.get_best_models(1)[0]
        model_filename = self._get_unique_filename("best_lstm_regressor.h5")
        best_model.save(model_filename)
        print(f"Best LSTM model saved to: {model_filename}")

        self.best_parameter = best_model
        print(f"Best parameters: {tuner.get_best_hyperparameters()[0].values}")

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_parameter.predict(X_test).flatten()

    def run_lstm_regressor(self, X_train, y_train, X_test, y_test, n_iter=10, epochs=10):
        print("\n" + "="*40)
        print(" Training LSTM Regressor ".center(40, "="))
        print("="*40)
        
        self.train_lstm(X_train, y_train, X_test, y_test, n_iter=n_iter, epochs=epochs)
        
        print("\n" + "="*40)
        print(" Evaluating LSTM Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
