from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform
import matplotlib.pyplot as plt
import numpy as np
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_LightGBM_Regressor:
    """
    LightGBM Regressor.
    
    How it works:
    A fast Gradient Boosting implementation.
    
    Why use it?
    - Faster training speed and higher efficiency.
    - Low memory usage.
    - Capable of handling large-scale data.
    """

    def __init__(self):
        self.best_parameter = None

    def train_lightgbm(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing LightGBM Regressor...")

        param_dist = {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 20),
            'learning_rate': loguniform(0.01, 0.2),
            'num_leaves': randint(20, 150),
            'min_child_samples': randint(5, 30),
            'subsample': loguniform(0.5, 1.0),
            'colsample_bytree': loguniform(0.5, 1.0)
        }

        model = LGBMRegressor(random_state=random_state)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.best_parameter = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV Score (Neg RMSE): {random_search.best_score_:.4f}")

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def plot_feature_importance(self, feature_names=None, top_n=10):
        """
        Visualizes the most important features.
        """
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        
        importances = self.best_parameter.feature_importances_
        indices = np.argsort(importances)[::-1]

        if feature_names is None:
            # Create dummy names if none provided
            feature_names = [f'Feature {i}' for i in range(len(importances))]

        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(top_n), importances[top_indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45)
        plt.xlabel("Variables")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()

    def run_lightgbm_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training LightGBM Regressor ".center(40, "="))
        print("="*40)
        
        self.train_lightgbm(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating LightGBM Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
        
        # Uncomment if you want to see feature importance automatically (requires feature names passed down)
        # self.plot_feature_importance()
