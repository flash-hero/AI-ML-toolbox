from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, DotProduct
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_GaussianProcess_Regressor:
    """
    Gaussian Process Regressor (GPR).
    
    How it works:
    A powerful probabilistic model. Instead of learning a single line, it learns a distribution over all possible functions that fit the data.
    It uses "Kernels" to define how similar different data points are.
    
    Why use it?
    - Provides uncertainty estimates for every prediction.
    - Very flexible (can model complex, non-linear patterns).
    - Best for small to medium-sized datasets (scales poorly with large data).
    """

    def __init__(self):
        self.best_gpr = None

    def train_gpr(self, X_train, y_train, n_iter=20, cv=5, random_state=42):
        print("Optimizing Gaussian Process Regressor...")
        print("This might take a while...")

        # Kernels define the "shape" and "smoothness" of the function we are looking for
        kernels = [
            ConstantKernel() * RBF(),                       # Standard smooth curve
            ConstantKernel() * RBF() + WhiteKernel(),       # Smooth curve + Noise
            ConstantKernel() * Matern(nu=1.5),              # Rougher curve
            ConstantKernel() * Matern(nu=2.5),              # Slightly smoother Matern
            ConstantKernel() * DotProduct() + WhiteKernel() # Linear-like trend
        ]

        param_dist = {
            'kernel': kernels,
            'alpha': [1e-10, 1e-8, 1e-6, 1e-4], # Noise level in observations
            'normalize_y': [True, False],       # Whether to scale the target variable
            'n_restarts_optimizer': sp_randint(0, 11) # How many times to retry finding the best fit
        }

        model = GaussianProcessRegressor(random_state=random_state)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(X_train, y_train)
        self.best_gpr = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        return self

    def predict(self, X_test):
        if self.best_gpr is None:
            raise ValueError("Model has not been trained yet.")
        print("Predicting...")
        return self.best_gpr.predict(X_test)

    def run_gaussian_process_regression_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training GPR Model ".center(40, "="))
        print("="*40)
        
        self.train_gpr(X_train, y_train)
        
        print("\n" + "="*40)
        print(" Evaluating GPR Model ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)

        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
