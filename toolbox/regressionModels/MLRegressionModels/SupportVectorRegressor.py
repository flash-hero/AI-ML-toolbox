import numpy as np
import random
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_SVR_Regressor:
    """
    Support Vector Regressor (SVR).
    
    How it works:
    Similar to SVM for classification, but for regression.
    It tries to find a function that deviates from the actual target values by a value no greater than epsilon 
    for each training point, and at the same time is as flat as possible.
    
    Why use it?
    - Effective in high dimensional spaces.
    - Robust to outliers (determined by C).
    """

    def __init__(self):
        self.best_svr = None
        self.best_params = None
        self.best_score = float('inf')

    def _random_sample(self):
        """
        Generates a valid random set of parameters for SVR.
        """
        C = np.random.uniform(0.1, 10)
        epsilon = np.random.uniform(0.01, 1)
        kernel = random.choice(['linear', 'rbf', 'poly'])
        gamma = random.choice(['scale', 'auto'])

        if kernel == 'poly':
            degree = random.choice([2, 3, 4])
            return {'C': C, 'epsilon': epsilon, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
        else:
            return {'C': C, 'epsilon': epsilon, 'kernel': kernel, 'gamma': gamma}

    def train_svr(self, X_train, y_train, n_iter=50, cv=5, random_state=42):
        """
        Custom loop for Hyperparameter Search (Manual implementation of Random Search).
        """
        print("Optimizing SVR with Custom Random Search...")
        
        np.random.seed(random_state)
        random.seed(random_state)

        for i in range(n_iter):
            params = self._random_sample()
            model = SVR(**params)
            try:
                # Cross-validation with scoring = Negative MSE (so lower is better in absolute terms, but higher is better in code)
                # Here we want to MINIMIZE error, so we track the lowest score?
                # The original code implemented: score = -np.mean(neg_mse). This makes score POSITIVE MSE.
                # So we want the LOWEST score.
                score = -np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))

                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self.best_svr = model

            except Exception as e:
                # print(f"Failed for params {params}: {e}")
                continue

        print(f"\nBest parameters: {self.best_params}")
        print(f"Best MSE (CV): {self.best_score:.4f}")

        # Retrain the best model on full data
        self.best_svr.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        if self.best_svr is None:
            raise ValueError("SVR model has not been trained.")
        print("Predicting...")
        return self.best_svr.predict(X_test)

    def run_svr_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training SVR Model ".center(40, "="))
        print("="*40)
        
        self.train_svr(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating SVR Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
