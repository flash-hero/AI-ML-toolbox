from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import uniform as sp_uniform
from evaluationModels.evaluation_classification import ClassifierEvaluator


class Method_NaiveBayes_Classifier:
    """
    Naive Bayes (Gaussian) Classifier.
    
    How it works:
    Based on Bayes' Theorem (Probability). 
    It assumes that all features are independent of each other (which is "Naive", because usually they aren't).
    For example, it assumes "Age" has nothing to do with "Income", which might be wrong, 
    but the model still works surprisingly well.
    
    Why use it?
    - Extremely fast.
    - Good for text classification and simple baselines.
    """

    def __init__(self):
        self.best_parameter = None
    
    def train_naive_bayes(self, X_train, y_train, n_iter=100, cv=5, random_state=42):
        print("Optimizing Naive Bayes...")
        
        nb = GaussianNB()
        
        # 'var_smoothing' is a stability calculation parameter (adding a tiny number to variance)
        param_dist = {
            'var_smoothing': sp_uniform(1e-10, 1e-9)
        }
        
        random_search = RandomizedSearchCV(nb, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        self.best_parameter = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")
        
        return self

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def run_naive_bayes_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Naive Bayes Model ".center(40, "="))
        print("="*40)
        
        self.train_naive_bayes(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Naive Bayes Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test) 
        
        evaluator = ClassifierEvaluator(y_test, y_pred) 
        evaluator.evaluation_metrics()
