from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_ExtraTrees_Regressor:
    """
    Extra Trees (Extremely Randomized Trees) Regressor.
    
    How it works:
    A variation of Random Forest. 
    It builds many trees, but instead of finding the *perfect* split at each step, 
    it picks random splits and chooses the best among them.
    
    Why use it?
    - Faster than Random Forest.
    - Can be more robust to noise and overfitting.
    """

    def __init__(self):
        self.best_parameter = None

    def train_extra_trees(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing Extra Trees Regressor...")

        param_dist = {
            'n_estimators': randint(100, 1000),       # Number of trees
            'max_depth': randint(5, 50),              # Depth
            'min_samples_split': randint(2, 20),      # Min samples to split
            'min_samples_leaf': randint(1, 20),       # Min samples in leaf
            'max_features': ['auto', 'sqrt', 'log2', None],
            'ccp_alpha': uniform(0.0, 0.02),          # Pruning parameter
            'max_leaf_nodes': randint(10, 200)
        }

        model = ExtraTreesRegressor(random_state=random_state)

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

    def run_extra_trees_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Extra Trees Regressor ".center(40, "="))
        print("="*40)
        
        self.train_extra_trees(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Extra Trees Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
