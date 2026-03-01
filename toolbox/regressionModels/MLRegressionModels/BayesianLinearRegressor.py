import numpy as np
import pymc as pm
import arviz as az
import random
import itertools
from sklearn.metrics import r2_score
from evaluationModels.evaluation_regressor import RegressionEvaluator

class Method_BayesianLinear_Regressor:
    """
    Bayesian Linear Regression (using PyMC).
    
    How it works:
    Instead of just finding one "best" line to fit the data (like standard Linear Regression),
    Bayesian Regression finds a *distribution* of possible lines.
    It treats the model parameters (slope and intercept) as probabilities, not fixed numbers.
    
    Why use it?
    - It gives you uncertainty estimates (e.g., "The price is likely 100, but could be between 90 and 110").
    - Good when you have prior knowledge about the data.
    - Robust to small datasets.
    """

    def __init__(self):
        self.best_params = None
        self.best_r2 = -np.inf
        self.model = None
        self.trace = None
        self.y_pred = None

    def train_bayesian(self, X_train, y_train, mu_values=[0, 1, -1], sigma_values=[1, 5, 10], n_iter=5):
        """
        Trains the model by sampling different prior beliefs (mu, sigma) and picking the best one.
        """
        # Create combinations of hyperparameters to test
        param_combinations = list(itertools.product(mu_values, sigma_values))
        # Randomly select a few to try (Random Search)
        sampled_params = random.sample(param_combinations, min(n_iter, len(param_combinations)))

        for mu, sigma in sampled_params:
            # Define the Probability Model
            with pm.Model() as model:
                # Priors for unknown model parameters
                coeffs = pm.Normal('coeffs', mu=mu, sigma=sigma, shape=X_train.shape[1])
                intercept = pm.Normal('intercept', mu=mu, sigma=sigma)
                sigma_noise = pm.HalfNormal('sigma', sigma=5)

                # Expected value of outcome
                mu_pred = pm.math.dot(X_train, coeffs) + intercept

                # Likelihood (sampling distribution) of observations
                Y_obs = pm.Normal('Y_obs', mu=mu_pred, sigma=sigma_noise, observed=y_train)

                # Inference: Draw samples from the posterior distribution
                # (This is the "Training" part where it learns from data)
                trace = pm.sample(500, tune=500, target_accept=0.9, progressbar=False, chains=1)

            # Evaluate this model
            # Retrieve learned parameters (average of the samples)
            a = trace.posterior['coeffs'].mean(dim=["chain", "draw"]).values
            b = trace.posterior['intercept'].mean(dim=["chain", "draw"]).values
            
            # Predict on training data to check performance
            y_pred_train = np.dot(X_train, a) + b
            r2 = r2_score(y_train, y_pred_train)

            # Save if it's the best so far
            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_params = {'mu': mu, 'sigma': sigma}
                self.trace = trace
                self.model = model

        print(f"Best parameters: {self.best_params}, R² (Train) = {self.best_r2:.4f}")

    def predict(self, X_test):
        if self.trace is None:
            raise ValueError("Model has not been trained yet.")
            
        # Use the mean of the learned parameters for prediction
        a = self.trace.posterior['coeffs'].mean(dim=["chain", "draw"]).values
        b = self.trace.posterior['intercept'].mean(dim=["chain", "draw"]).values
        
        y_pred = np.dot(X_test, a) + b
        self.y_pred = y_pred
        return y_pred

    def run_bayesian_linear_regressor(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*40)
        print(" Training Bayesian Linear Regression ".center(40, "="))
        print("="*40)
        
        self.train_bayesian(X_train, y_train)

        print("\n" + "="*40)
        print(" Evaluating Bayesian Linear Regression ".center(40, "="))
        print("="*40)
        
        y_pred = self.predict(X_test)
        
        evaluator = RegressionEvaluator(y_test, y_pred)
        evaluator.evaluation_metrics()
