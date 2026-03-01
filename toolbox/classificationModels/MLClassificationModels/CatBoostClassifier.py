from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from scipy.stats import randint, uniform
import numpy as np
from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_CatBoost_Classifier:
    """
    CatBoost (Categorical Boosting) Classifier.
    
    How it works:
    It is a Gradient Boosting algorithm (like XGBoost or LightGBM) but specifically optimized for
    Categorical Data (text categories).
    
    Why use it?
    - It handles categorical variables automatically (no manual encoding needed usually, though we often do it anyway).
    - It is very fast and accurate.
    - It reduces overfitting.
    """
    
    def __init__(self):
        self.best_parameter = None

    def train_catboost(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing hyperparameters with RandomizedSearchCV...")

        # Check if we are doing Binary (2 classes) or Multi-class classification
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            loss_function = 'Logloss'
        else:
            loss_function = 'MultiClass'

        # Parameters to tune
        param_dist = {
            'iterations': randint(100, 1000),      # Number of trees
            'depth': randint(4, 10),               # Depth of trees
            'learning_rate': uniform(0.01, 0.3),   # Speed of learning
            'l2_leaf_reg': uniform(1, 10),         # Regularization (prevents overfitting)
            'border_count': randint(32, 255),      # Complexity of splits
            'bagging_temperature': uniform(0, 1),
            'random_strength': uniform(0, 1),
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
        }

        # Initialize CatBoost
        # verbose=0 means be silent (don't print a log line for every tree)
        model = CatBoostClassifier(
            loss_function=loss_function,
            eval_metric='Accuracy',
            random_state=random_state,
            verbose=0
        )

        # Find best parameters
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

    def run_catboost_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training CatBoost Model ".center(40, "="))
        print("="*40)
        
        self.train_catboost(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating CatBoost Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
