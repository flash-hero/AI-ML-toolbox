from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score

from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_Bagging_Classifier:
    """
    Bagging (Bootstrap Aggregating) Classifier.
    
    How it works:
    1. It creates many copies of the training data by random sampling (with replacement).
       (Imagine taking 5 different random surveys from the same population).
    2. It trains a separate model (usually a Decision Tree) on each copy.
    3. Each model makes a prediction.
    4. The final prediction is the "Correction Majority Vote" (for classification) or Average (for regression).
    
    Why use it?
    It reduces variance. If one individual tree is crazy or over-sensitive to noise, 
    averaging it with many others cancels out the noise.
    """

    def __init__(self):
        self.best_estimator_ = None

    def train(self, X, y, n_iter=20, cv=5, rs=42):
        """
        Trains the Bagging model using Randomized Search.
        """
        print("Optimizing hyperparameters for Bagging...")
        
        # We start with a Bagging algorithm that uses Decision Trees inside it.
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=rs),
            random_state=rs
        )

        # Parameters to test
        param_dist = {
            "estimator__max_depth": randint(1, 8),            # Max tree depth (shallow trees avoid overfitting)
            "estimator__min_samples_split": randint(2, 10),   # Min samples needed to split a node
            "n_estimators": randint(10, 150),                 # Number of trees in the bag
            "max_samples": uniform(0.5, 0.5),                 # Fraction of original data to use for each tree (0.5 to 1.0)
            "max_features": uniform(0.5, 0.5),                # Fraction of features (columns) to use for each tree
            "bootstrap": [True, False],                       # Whether to sample with replacement
        }

        # Search for best parameters
        search = RandomizedSearchCV(
            bag, param_dist,
            n_iter=n_iter, cv=cv,
            scoring=make_scorer(accuracy_score),
            n_jobs=-1, random_state=rs, verbose=0
        )
        
        search.fit(X, y)
        self.best_estimator_ = search.best_estimator_
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best Cross-Validation Score: {search.best_score_:.4f}")

    def predict(self, X):
        if self.best_estimator_ is None:
             raise ValueError("Model has not been trained yet.")
        return self.best_estimator_.predict(X)

    def run_bagging_classifier(self, X_train, y_train, X_test, y_test):
        """
        Main runner function: Trains, Predicts, and Evaluates.
        """
        print("\n" + "="*40)
        print(" Training Bagging Model ".center(40, "="))
        print("="*40)
        
        self.train(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Bagging Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
