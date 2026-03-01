from sklearn.linear_model import LinearRegression
from evaluationModels.evaluation_regressor import RegressionEvaluator


class Method_Linear_Regressor:
    """
    Linear Regression.
    
    How it works:
    The most basic and fundamental algorithm. It tries to draw a straight line (or plane) 
    that passes as close as possible to all data points.
    Equation: y = mx + c (or y = w1*x1 + w2*x2 + ... + b)
    
    Why use it?
    - Simple and fast.
    - Easy to interpret (we know exactly how much each feature increases the price/score).
    - Good baseline to compare other complex models against.
    """

    def __init__(self):
        # Attribute to save the trained model
        self.model = None

    def train_linear_regression(self, X_train, y_train):
        print("Training Multiple Linear Regression Model...")

        model = LinearRegression()
        model.fit(X_train, y_train)

        self.model = model
        print("Model trained successfully.")
        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting with trained model...")
        return self.model.predict(X_test)

    def run_linear_regression_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Multiple Linear Regression ".center(40, "="))
        print("="*40)

        self.train_linear_regression(X_train, y_train)

        y_pred = self.predict(X_test)

        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
