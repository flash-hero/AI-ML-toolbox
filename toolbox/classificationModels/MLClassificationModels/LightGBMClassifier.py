from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score

from evaluationModels.evaluation_classification import ClassifierEvaluator

import warnings
warnings.filterwarnings("ignore")

class Method_LightGBM_Classifier:
    """
    LightGBM (Light Gradient Boosting Machine) Classifier.
    
    How it works:
    Another Gradient Boosting algorithm (like XGBoost).
    
    Why use it?
    - "Light" refers to speed and memory usage. It is extremely fast.
    - It builds trees differently (leaf-wise instead of level-wise), which often leads to better accuracy.
    """

    def __init__(self):
        self.best_parameter = None

    def train_lgbm(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing LightGBM with RandomizedSearchCV...")

        # Parameters to tune
        param_dist = {
            'learning_rate': uniform(0.01, 0.2), # Speed of learning
            'max_depth': randint(3, 8),          # Tree depth
            'num_leaves': randint(2, 32),        # Max leaves per tree
            'min_data_in_leaf': randint(1, 10),  # Min samples in a leaf
            'min_data_in_bin': randint(1, 10),   # Optimization parameter for binning
            'n_estimators': randint(50, 150)     # Number of trees
        }

        model = LGBMClassifier()

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

    def run_lightgbm_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training LightGBM Model ".center(40, "="))
        print("="*40)
        
        self.train_lgbm(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating LightGBM Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
