from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from evaluationModels.evaluation_regressor import RegressionEvaluator
from scipy.stats import uniform as sp_uniform, randint as sp_randint

class Method_DecisionTree:
    """
    Decision Tree Regressor.
    
    How it works:
    Splits the data into smaller groups based on simple rules (e.g., "If House Size > 1000").
    For each group, it predicts the *average* value of the target variable in that group.
    
    Why use it?
    - Easy to visualize and understand.
    - Captures non-linear relationships well.
    """

    def __init__(self):
        self.best_parameter = None

    def train_decision_tree(self, X_train, y_train, n_iter=100, cv=5, random_state=42):
        print("Optimizing Decision Tree Regressor...")
        
        dt = DecisionTreeRegressor()
        
        # Parameters to tune
        param_dist = {
            'max_depth': sp_randint(1, 20),
            'min_samples_split': sp_randint(2, 20),
            'min_samples_leaf': sp_randint(1, 20),
            'max_features': sp_uniform(0.1, 0.9)
        }
        
        random_search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        self.best_parameter = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")

        return self

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def run_decision_tree_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Decision Tree Regressor ".center(40, "="))
        print("="*40)
        
        self.train_decision_tree(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Decision Tree Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test) 
        
        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
