from sklearn.linear_model import BayesianRidge
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from evaluationModels.evaluation_regressor import RegressionEvaluator 


class Method_BayesianRidge_Regressor:
    """
    Bayesian Ridge Regression.
    
    How it works:
    Similar to standard Ridge Regression (which penalizes large coefficients to prevent overfitting),
    but formulated in a Bayesian framework.
    It automatically estimates the regularization parameters from the data.
    
    Why use it?
    - Auto-tuning of regularization (less manual work).
    - Robust to overfitting.
    """

    def __init__(self):
        self.best_parameter = None

    def train_bayesian(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing Bayesian Ridge...")

        # Parameters for the Gamma prior distributions (controlling regularization)
        param_dist = {
            'alpha_1': uniform(1e-6, 1e-3),
            'alpha_2': uniform(1e-6, 1e-3),
            'lambda_1': uniform(1e-6, 1e-3),
            'lambda_2': uniform(1e-6, 1e-3)
        }

        model = BayesianRidge()

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.best_parameter = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV Score (Neg MSE): {random_search.best_score_:.4f}")

        return self

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def run_bayesian_ridge_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Bayesian Ridge Model ".center(40, "="))
        print("="*40)
        
        self.train_bayesian(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Bayesian Ridge Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = RegressionEvaluator(y_test, y_pred)  
        evaluator.evaluation_metrics()