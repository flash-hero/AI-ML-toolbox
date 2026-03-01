import numpy as np
from sklearn.svm import SVC
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_SVM_Classifier:
    """
    Support Vector Machine (SVM) Classifier.
    
    How it works:
    It tries to find the best boundary (hyperplane) that separates the classes with the widest possible "street" (margin).
    If the data can't be separated by a straight line, it uses "Kernels" to project the data into higher dimensions 
    where it *can* be separated.
    
    Why use it?
    - Very accurate for small to medium complex datasets.
    - Works well when there are many features (columns).
    """

    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def train_svm(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing SVM...")

        # Parameters to tune
        param_dist = {
            'C': uniform(0.1, 10),              # "Strictness" (High C = don't miss any point, Low C = allow some errors/smoother boundary)
            'kernel': ['linear', 'rbf', 'poly'],# Transformation type (Linear = straight line, RBF/Poly = curved boundaries)
            'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)), # How far the influence of a single example reaches
            'degree': [2, 3, 4]                 # Degree for polynomial kernel
        }

        base_model = SVC(random_state=random_state)

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=make_scorer(accuracy_score),
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_

        print(f"Best parameters: {self.best_params}")
        print(f"Best Cross-Validation Score: {self.best_score:.4f}")
        return self

    def predict(self, X_test):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_model.predict(X_test)

    def run_svm_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training SVM Model ".center(40, "="))
        print("="*40)
        
        self.train_svm(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating SVM Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
