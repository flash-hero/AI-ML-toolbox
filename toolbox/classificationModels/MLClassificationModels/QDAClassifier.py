from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from scipy.stats import uniform
from evaluationModels.evaluation_classification import ClassifierEvaluator


class Method_QDA_Classifier:
    """
    Quadratic Discriminant Analysis (QDA) Classifier.
    
    How it works:
    Similar to LDA (Linear Discriminant Analysis), but it allows for curved boundaries between classes 
    instead of just straight lines.
    
    Why use it?
    - When the data is not separable by a straight line.
    - Requires more data than LDA to estimate the curves properly.
    """

    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def train_qda(self, X_train, y_train, n_iter=10, cv=5, random_state=42):
        print("Optimizing QDA with RandomizedSearchCV...")

        # Regularization parameter
        param_dist = {
            'reg_param': uniform(0, 1)
        }

        model = QuadraticDiscriminantAnalysis()

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

    def run_qda_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training QDA Model ".center(40, "="))
        print("="*40)
        
        self.train_qda(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating QDA Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
