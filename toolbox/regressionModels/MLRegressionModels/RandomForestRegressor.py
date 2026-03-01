from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import uniform as sp_uniform, randint as sp_randint
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_RandomForest:
    """
    Random Forest Regressor.
    
    How it works:
    An ensemble of many Decision Trees.
    Each tree predicts a value, and the final prediction is the average of all trees.
    
    Why use it?
    - Very robust and accurate.
    - "Power of the crowd": Averaging many errors cancels them out, leading to a stable prediction.
    """

    def __init__(self):
        self.best_parameter = None

    def train_random_forest(self, X_train, y_train, n_iter=100, cv=5, random_state=42):
        print("Optimizing Random Forest Regressor...")
        
        rf = RandomForestRegressor()
        
        param_dist = {
            'n_estimators': sp_randint(10, 200),      # Number of trees
            'max_depth': sp_randint(1, 20),           # Max depth
            'min_samples_split': sp_randint(2, 20),   # Min samples to split
            'min_samples_leaf': sp_randint(1, 20),    # Min samples in leaf
            'max_features': sp_uniform(0.1, 0.9)      # Fraction of features to consider
        }
        
        random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        self.best_parameter = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")

        return self

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def run_random_forest_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Random Forest Regressor ".center(40, "="))
        print("="*40)
        
        self.train_random_forest(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Random Forest Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test) 
        
        evaluator = RegressionEvaluator(y_test, y_pred) 
        evaluator.evaluation_metrics() 
