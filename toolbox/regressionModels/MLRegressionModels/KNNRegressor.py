from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint as sp_randint
from evaluationModels.evaluation_regressor import RegressionEvaluator 

class Method_KNN_Regressor:
    """
    K-Nearest Neighbors (KNN) Regressor.
    
    How it works:
    To predict the value for a new data point, it looks at the 'K' closest data points 
    in the training set and calculates their average.
    
    Why use it?
    - Simple and intuitive.
    - Non-parametric (makes no assumptions about the data distribution).
    """

    def __init__(self):
        self.best_knn = None
    
    def train_knn(self, X_train, y_train, n_iter=100, cv=5, random_state=42):
        print("Optimizing KNN Regressor...")

        knn = KNeighborsRegressor()
        
        param_dist = {
            'n_neighbors': sp_randint(1, 25),              # Number of neighbors to Average
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'] # Distance metric
        }

        random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)

        self.best_knn = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")

        return self

    def predict(self, X_test):
        if self.best_knn is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting...")
        return self.best_knn.predict(X_test)

    def run_knn_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training KNN Regressor ".center(40, "="))
        print("="*40)
        
        self.train_knn(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating KNN Regressor ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
