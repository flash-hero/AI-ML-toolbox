from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform

from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_ExtraTrees_Classifier:
    """
    Extra Trees (Extremely Randomized Trees) Classifier.
    
    How it works:
    It is very similar to Random Forest (a team of Decision Trees).
    The main difference is that Extra Trees are even more random:
    - Random Forest: Searches for the BEST possible rule to split data at each step.
    - Extra Trees: Picks a few random rules and just chooses the best among those (without looking at everything).
    
    Why use it?
    - It's often faster than Random Forest.
    - Sometimes, being "less perfect" at each step helps avoid over-memorizing the data (reduces overfitting).
    """

    def __init__(self):
        self.best_parameter = None

    def train_extra_trees(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing Extra Trees...")

        # Parameters to tune
        param_dist = {
            'n_estimators': randint(100, 1000),       # Number of trees
            'max_depth': randint(5, 50),              # Max depth of trees
            'min_samples_split': randint(2, 20),      # Min samples to split
            'min_samples_leaf': randint(1, 20),       # Min samples at leaf
            'max_features': ['auto', 'sqrt', 'log2', None], # How many features to look at
            'ccp_alpha': uniform(0.0, 0.02),          # Complexity parameter (pruning)
            'max_leaf_nodes': randint(10, 200)        # Max number of end-nodes
        }

        model = ExtraTreesClassifier(random_state=random_state)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='accuracy',
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.best_parameter = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best Cross-Validation Score: {random_search.best_score_:.4f}")

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def run_extra_trees_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Extra Trees Model ".center(40, "="))
        print("="*40)
        
        self.train_extra_trees(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Extra Trees Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
