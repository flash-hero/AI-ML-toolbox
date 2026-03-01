from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from evaluationModels.evaluation_classification import ClassifierEvaluator


class Method_LogisticRegression_Classifier:
    """
    Logistic Regression Classifier.
    
    How it works:
    Despite the name "Regression", this is used for Classification (Yes/No, True/False).
    It calculates the *probability* that an example belongs to a class using a curve (sigmoid function).
    If probability > 50%, it says "Yes".
    
    Why use it?
    - It's simple, fast, and easy to interpret (we can see which coefficients affect the outcome).
    - Great baseline model.
    """

    def __init__(self):
        self.best_parameter = None

    def train_logistic(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing Logistic Regression...")

        # Parameters to tune
        param_dist = {
            'C': loguniform(1e-4, 1e4),   # Inverse of regularization strength (smaller = stronger regularization)
            'penalty': ['l1', 'l2'],      # Regularization type (Lasso vs Ridge)
            'solver': ['liblinear'],      # Algorithm suitable for small datasets and L1 penalty
            'max_iter': [1000]            # Maximum number of steps to find the solution
        }

        model = LogisticRegression()

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

    def run_logistic_regression_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Logistic Regression Model ".center(40, "="))
        print("="*40)
        
        self.train_logistic(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Logistic Regression Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()