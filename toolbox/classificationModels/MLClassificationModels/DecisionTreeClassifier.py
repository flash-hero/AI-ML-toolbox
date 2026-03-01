from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import uniform as sp_uniform, randint as sp_randint

from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_DecisionTree_Classifier:
    """
    Decision Tree Classifier.
    
    How it works:
    Imagine a flowchart. 
    - "Is it raining?" -> Yes -> "Is it windy?" -> No -> "Go Outside".
    - "Is it raining?" -> No -> "Stay Inside".
    
    The algorithm learns these rules automatically from the data to classify new examples.
    It splits the data into smaller and smaller groups until each group is mostly one class.
    """

    def __init__(self):
        self.best_parameter = None

    def train_decision_tree(self, X_train, y_train, n_iter=100, cv=5, random_state=42):
        print("Optimizing Decision Tree...")
        dt = DecisionTreeClassifier()
        
        # Parameters to tune
        param_dist = {
            'max_depth': sp_randint(1, 20),           # How deep the tree can grow (too deep = overfitting)
            'min_samples_split': sp_randint(2, 20),   # Min samples required to create a new branch
            'min_samples_leaf': sp_randint(1, 20),    # Min samples required at the end of a branch (leaf)
            'max_features': sp_uniform(0.1, 0.9)      # Percentage of features to consider for each split
        }
        
        random_search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        self.best_parameter = random_search.best_estimator_
        print(f"Best parameters found: {random_search.best_params_}")

        return self

    def predict(self, X_test):
        if self.best_parameter is None:
            raise ValueError("Model has not been trained yet.")
        
        print("Predicting...")
        return self.best_parameter.predict(X_test)

    def run_decision_tree_classifier(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Decision Tree Model ".center(40, "="))
        print("="*40)
        
        self.train_decision_tree(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating Decision Tree Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        # Calculate and show Accuracy and other metrics
        # (Accuracy calculation here is redundant because evaluator shows it too, but kept for clarity)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Quick Accuracy Check: {accuracy * 100:.2f}%')
        
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
