from sklearn.ensemble import AdaBoostClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score 
from scipy.stats import randint, uniform

# Import our custom evaluator
from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_AdaBoost_Classifier:
    """
    AdaBoost (Adaptive Boosting) Classifier.
    
    How it works:
    1. It starts by training a simple model (a "weak learner", usually a small Decision Tree).
    2. It sees which examples the model got wrong.
    3. It trains a *new* model that focuses specifically on those hard-to-predict examples.
    4. It repeats this process many times.
    5. Final prediction is a vote from all these models (weighted by their accuracy).
    
    Think of it as a team of students: if one student is bad at math, we add a math expert to the team.
    If another is bad at history, we add a history expert. Together, the team is smart.
    """

    def __init__(self):
        """
        Initializes the method.
        We will store the best trained model here after optimization.
        """
        self.best_model = None

    def train_adaboost(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        """
        Trains the AdaBoost model using Randomized Search to find the best settings.
        
        Parameters:
        - X_train, y_train: Training data and labels.
        - n_iter: Number of different settings to try.
        - cv: Number of cross-validation folds (how many times we allow the model to practice).
        """
        print("Optimizing hyperparameters with RandomizedSearchCV...")

        # Define the "playground" for parameters - the computer will try random combinations from here
        param_dist = {
            'n_estimators': randint(50, 300),       # How many weak models (students) to add to the team
            'learning_rate': uniform(0.01, 1.0)     # How much each model contributes (lower = slower but often better)
        }

        # The basic unit of our team: a "Stump" (a Decision Tree with only 1 split)
        base_learner = DecisionTreeClassifier(max_depth=1)

        # Initialize the AdaBoost algorithm with our base learner
        model = AdaBoostClassifier(estimator=base_learner, random_state=random_state)

        # Set up the search for the best configuration
        random_search = RandomizedSearchCV(
            estimator=model,                        # The model to tune
            param_distributions=param_dist,         # The parameters to test
            n_iter=n_iter,                          # Number of tries
            scoring=make_scorer(accuracy_score),    # Goal: Maximize Accuracy
            cv=cv,                                  # Cross-validation folds
            random_state=random_state,              # For reproducibility
            n_jobs=-1                               # Use all CPU cores for speed
        )

        # Start Training
        random_search.fit(X_train, y_train)

        # Save the winner
        self.best_model = random_search.best_estimator_

        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best validation score: {random_search.best_score_:.4f}")
        return self

    def predict(self, X_test):
        """
        Generates predictions using the best trained model.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")

        print("Predicting with the optimal model...")
        return self.best_model.predict(X_test)

    def run_adaboost_classifier(self, X_train, y_train, X_test, y_test):
        """
        Main runner function: Trains, Predicts, and Evaluates the model.
        """
        print("\n" + "="*40)
        print(" Training AdaBoost Model ".center(40, "="))
        print("="*40)
        
        self.train_adaboost(X_train, y_train)

        print("\n" + "="*40)
        print(" Evaluating AdaBoost Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        # Use our custom evaluator to show metrics (Accuracy, F1, Confusion Matrix)
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
