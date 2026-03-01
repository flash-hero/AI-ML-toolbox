from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score
from scipy.stats import randint, uniform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluationModels.evaluation_regressor import RegressionEvaluator 

class Method_GradientBoosting_Regressor:
    """
    Gradient Boosting Regressor.
    
    How it works:
    Builds an additive model in a forward stage-wise fashion.
    It allows for the optimization of arbitrary differentiable loss functions.
    Basically: It makes a guess, sees how far off it is, and then makes a new guess to fix the error.
    
    Why use it?
    - Extremely effective for regression tasks.
    - Can handle different types of data distribution.
    """

    def __init__(self):
        self.best_parameter = None

    def train_gbr(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing Gradient Boosting Regressor...")

        param_dist = {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.6, 0.4),
            'alpha': uniform(0.1, 0.9)
        }

        model = GradientBoostingRegressor(random_state=random_state)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=make_scorer(r2_score),
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(X_train, y_train)
        self.best_parameter = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV Score (R2): {random_search.best_score_:.4f}")
        return self

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def feature_importance(self, feature_names):
        """
        Visualizes which features were most important for the prediction.
        """
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")

        importances = self.best_parameter.feature_importances_

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

        return feature_importance_df

    def run_gradient_boosting_regressor(self, X_train, y_train, X_test, y_test, feature_names=None):
        print("\n" + "="*40)
        print(" Training Gradient Boosting ".center(40, "="))
        print("="*40)
        
        self.train_gbr(X_train, y_train)

        print("\n" + "="*40)
        print(" Evaluating Gradient Boosting ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()

        # Optional: Plot feature importance if feature names are provided
        if feature_names is not None:
             print("\n" + "="*40)
             print(" Feature Importance ".center(40, "="))
             print("="*40)
             self.feature_importance(feature_names)