from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform
from evaluationModels.evaluation_regressor import RegressionEvaluator 

class Method_Catboost_Regressor:
    """
    CatBoost Regressor.
    
    How it works:
    Gradient Boosting optimized for categorical features. Provides high accuracy predictions.
    
    Why use it?
    - Handles categorical columns well (though we often encode them first).
    - Very stable and accurate.
    """

    def __init__(self):
        self.best_parameter = None

    def train_catboost(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing CatBoost Regressor...")

        # Parameters to tune
        param_dist = {
            'iterations': randint(100, 1001),        # Number of trees
            'depth': randint(4, 11),                 # Tree depth
            'learning_rate': loguniform(0.01, 0.2),  # Learning speed
            'l2_leaf_reg': loguniform(1, 10),        # Regularization
            'random_strength': randint(1, 4),        # Randomness for scoring splits
            'bagging_temperature': [0, 0.5, 1, 2]    # Bayesian Bootstrap settings
        }

        # Initialize CatBoost
        # verbose=0 to keep it quiet
        model = CatBoostRegressor(
            loss_function='RMSE',
            random_seed=random_state,
            verbose=0
        )

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

    def run_catboost_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training CatBoost Regressor ".center(40, "="))
        print("="*40)
        
        self.train_catboost(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating CatBoost Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
