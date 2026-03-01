from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import uniform, randint as sp_randint
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_XGBoost_Regressor:
    """
    XGBoost Regressor.
    
    How it works:
    Extreme Gradient Boosting. High-performance implementation of gradient boosted decision trees.
    
    Why use it?
    - Speed and performance.
    - Can harness the power of parallel processing.
    """

    def __init__(self):
        self.best_model = None

    def train_xgboost(self, X_train, y_train, n_iter=30, cv=5, random_state=42):
        print("Optimizing XGBoost Regressor...")

        model = XGBRegressor(objective='reg:squarederror', verbosity=0, random_state=random_state)

        # Reduced grid for speed
        param_dist = {
            'n_estimators': sp_randint(50, 200),
            'max_depth': sp_randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.6, 0.4)
        }

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=random_state
        )

        random_search.fit(X_train, y_train)
        self.best_model = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        return self

    def predict(self, X_test):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting with XGBoost...")
        return self.best_model.predict(X_test)

    def run_xgboost_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training XGBoost Regressor ".center(40, "="))
        print("="*40)
        
        self.train_xgboost(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating XGBoost Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()