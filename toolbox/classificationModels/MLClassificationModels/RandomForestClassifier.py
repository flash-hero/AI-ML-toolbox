from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from evaluationModels.evaluation_classification import ClassifierEvaluator


class Method_RandomForest:
    """
    Random Forest Classifier.
    
    How it works:
    A "Forest" is made of many Decision Trees.
    - Each tree is trained on a random part of the data.
    - Each tree votes for a class.
    - The class with the most votes wins.
    
    Why use it?
    - One of the most powerful and versatile algorithms.
    - Very resistant to overfitting (because of the voting system).
    """

    def __init__(self):
        # Fixed: Was _init_ in original code, changed to __init__
        self.best_rf = None

    def train_rf(self, X_train, y_train, param_dist=None, n_iter=100, cv=5, random_state=42):
        print("Optimizing Random Forest...")

        rf = RandomForestClassifier()
        
        if param_dist is None:
            # Parameters to tune
            param_dist = {
                'n_estimators': sp_randint(50, 500),      # Number of trees
                'max_depth': sp_randint(1, 20),           # Max depth per tree
                'min_samples_split': sp_randint(2, 20),   # Min samples to split
                'min_samples_leaf': sp_randint(1, 20),    # Min samples in leaf
                'max_features': ['sqrt', 'log2'],         # How many features to check
                'bootstrap': [True, False]                # Sampling method
            }

        random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)

        self.best_rf = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")
        return self

    def predict(self, X_test):
        if self.best_rf is None:
            raise ValueError("Model has not been trained yet.")
        
        print("Predicting...")
        return self.best_rf.predict(X_test)

    def run_random_forest_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Random Forest Model ".center(40, "="))
        print("="*40)
        
        self.train_rf(X_train, y_train)

        # Cross-validation check
        scores = cross_val_score(self.best_rf, X_train, y_train, cv=5)
        print(f"Validation Accuracy Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

        print("\n" + "="*40)
        print(" Evaluating Random Forest Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
