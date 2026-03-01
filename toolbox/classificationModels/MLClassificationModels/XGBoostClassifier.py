from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from xgboost import XGBClassifier
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from evaluationModels.evaluation_classification import ClassifierEvaluator


class Method_XGBoost_Classifier:
    """
    XGBoost (Extreme Gradient Boosting) Classifier.
    
    How it works:
    A highly optimized Gradient Boosting algorithm.
    It builds trees sequentially, where each new tree corrects the errors of the previous ones.
    
    Why use it?
    - It is often the "weapon of choice" for winning Machine Learning competitions (Kaggle).
    - Extremely accurate and efficient.
    - Handles missing values automatically.
    """

    def __init__(self):
        self.best_parameter = None

    def train_xgboost(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing XGBoost with RandomizedSearchCV...")

        # Parameters to tune
        param_dist = {
            'n_estimators': sp_randint(50, 200),        # Number of trees
            'max_depth': sp_randint(3, 10),             # Depth of trees
            'learning_rate': sp_uniform(0.01, 0.3),     # Learning speed
            'subsample': sp_uniform(0.7, 0.3),          # Fraction of samples per tree
            'colsample_bytree': sp_uniform(0.6, 0.4),   # Fraction of features per tree
            'gamma': sp_uniform(0, 5)                   # Regularization
        }

        # Initialize XGBoost
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            random_state=random_state,
            tree_method='auto', 
            device='cpu'         # Force CPU usage (safer compatibility)
        )

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=make_scorer(accuracy_score),
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
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

    def run_xgboost_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training XGBoost Model ".center(40, "="))
        print("="*40)
        
        self.train_xgboost(X_train, y_train)

        print("\n" + "="*40)
        print(" Evaluating XGBoost Model ".center(40, "="))
        print("="*40)

        y_pred = self.predict(X_test)

        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
