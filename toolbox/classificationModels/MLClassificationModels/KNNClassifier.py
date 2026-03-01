from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score 
from scipy.stats import randint as sp_randint

from evaluationModels.evaluation_classification import ClassifierEvaluator


class Method_KNN_Classifier:
    """
    K-Nearest Neighbors (KNN) Classifier.
    
    How it works:
    "Tell me who your friends are, and I'll tell you who you are."
    1. For a new data point, it looks for the 'K' closest data points in the training set.
    2. If most neighbors are Class A, it predicts Class A.
    
    K is just a number (e.g., 3, 5, 10).
    """

    def __init__(self):
        self.best_parameter = None

    def train_knn(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        """
        Trains the KNN model using Randomized Search.
        """
        print("Optimizing KNN Parameters...")

        # Parameters to tune
        param_dist = {
            'n_neighbors': sp_randint(1, 26),               # Try K between 1 and 25
            'metric': ['euclidean', 'manhattan',            # How to measure "distance"
                       'chebyshev', 'minkowski']
        }

        model = KNeighborsClassifier()

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=make_scorer(accuracy_score),
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.best_parameter = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best Cross-Validation Score: {random_search.best_score_:.4f}")

        return self

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting with optimal model...")
        return self.best_parameter.predict(X_test)

    def run_knn_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training KNN Model ".center(40, "="))
        print("="*40)
        
        self.train_knn(X_train, y_train)

        print("\n" + "="*40)
        print(" Evaluating KNN Model ".center(40, "="))
        print("="*40)

        y_pred = self.predict(X_test)

        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
