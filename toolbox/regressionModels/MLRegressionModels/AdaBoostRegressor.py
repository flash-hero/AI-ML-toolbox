from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import randint, uniform

from evaluationModels.evaluation_regressor import RegressionEvaluator 

class Method_AdaBoost_Regressor:
    """
    AdaBoost (Adaptive Boosting) Regressor.
    
    How it works:
    Similar to the Classifier version, but for predicting numbers (prices, temperatures, etc.).
    It builds a sequence of simple models (Decision Trees). 
    Each new model focuses on correcting the errors (difference between predicted and actual value) of the previous models.
    
    Why use it?
    - Combines many "weak" (simple) models to create a "strong" (accurate) one.
    - Improving performance step-by-step.
    """

    def __init__(self):
        self.best_model = None

    def train_adaboost(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing AdaBoost Regressor...")

        # Parameters to tune
        param_dist = {
            'n_estimators': randint(50, 300),     # Number of trees
            'learning_rate': uniform(0.01, 1.0)   # How much each tree contributes
        }

        # Base model: A small decision tree (depth 3)
        base_learner = DecisionTreeRegressor(max_depth=3)

        model = AdaBoostRegressor(estimator=base_learner, random_state=random_state)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            # We want to MINIMIZE error, but sklearn maximizes score. 
            # So we use negative MSE (Calculated as -Error). The highest negative number (closest to 0) is best.
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.best_model = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        # best_score_ is negative MSE here
        print(f"Best CV Score (Negative MSE): {random_search.best_score_:.4f}")
        return self

    def predict(self, X_test):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting...")
        return self.best_model.predict(X_test)

    def run_adaboost_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training AdaBoost Regressor ".center(40, "="))
        print("="*40)
        
        self.train_adaboost(X_train, y_train)

        print("\n" + "="*40)
        print(" Evaluating AdaBoost Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
