import sys
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import randint as sp_randint

from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_LDA_Classifier:
    """
    Linear Discriminant Analysis (LDA) Classifier.
    
    How it works:
    It tries to find a line (or plane) that best separates the different classes.
    It looks for a direction where the classes are far apart from each other, 
    but tightly clustered within themselves.
    """

    def __init__(self):
        self.best_lda = None
    
    def train_lda(self, X_train, y_train, n_iter=100, cv=5, random_state=42):
        print("Optimizing LDA Parameters...")

        lda = LinearDiscriminantAnalysis()
        
        # Parameters to tune
        param_dist = {
            'solver': ['svd', 'lsqr', 'eigen'], # Algorithms to solve the math
            'shrinkage': ['auto', None],        # Regularization (helps if few data points)
            # n_components usually limited by number of classes - 1. We test a few values.
            'n_components': [None, 2, 3, 4, 5] 
        }

        random_search = RandomizedSearchCV(lda, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)

        self.best_lda = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")

        return self

    def predict(self, X_test):
        if self.best_lda is None:
            raise ValueError("Model has not been trained yet.")
        
        print("Predicting...")
        return self.best_lda.predict(X_test)

    def run_lda_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training LDA Model ".center(40, "="))
        print("="*40)
        
        self.train_lda(X_train, y_train)

        print("\n" + "="*40)
        print(" Evaluating LDA Model ".center(40, "="))
        print("="*40)

        y_pred = self.predict(X_test)

        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
