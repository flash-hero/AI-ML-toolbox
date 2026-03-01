import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

class RegressionEvaluator:
    """
    This class evaluates the performance of a Regression model.
    It compares the 'Predicted' numbers with the 'Actual' numbers to see how close they are.
    """
    
    def __init__(self, y_true, y_pred):
        """
        Initializes the evaluator.
        
        Parameters:
        - y_true: The actual values (ground truth).
        - y_pred: The values predicted by the model.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluation_metrics(self, save_dir=None):
        """
        Calculates metrics and displays plots to visualize errors.
        """
        import os
        
        print("\n----- Regression Evaluation Report -----")

        # --- Calculate Metrics ---
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        evs = explained_variance_score(self.y_true, self.y_pred)

        # --- Print Metrics ---
        print(f"Mean Squared Error (MSE)          : {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE)    : {rmse:.4f}")
        print(f"Mean Absolute Error (MAE)         : {mae:.4f}")
        print(f"Coefficient of Determination (R²) : {r2:.4f}")
        print(f"Explained Variance Score          : {evs:.4f}")

        plots = {}

        # --- Plot 1: Predictions vs Actual ---
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_true, self.y_pred, alpha=0.7)
        plt.plot([min(self.y_true), max(self.y_true)], [min(self.y_true), max(self.y_true)], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs Actual')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
             path = os.path.join(save_dir, "pred_vs_actual.png")
             plt.savefig(path)
             plt.close()
             plots["pred_vs_actual"] = path
        else:
             plt.show()

        # --- Plot 2: Error Distribution ---
        errors = self.y_pred - self.y_true
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True)
        
        if save_dir:
             path = os.path.join(save_dir, "error_distribution.png")
             plt.savefig(path)
             plt.close()
             plots["error_distribution"] = path
        else:
             plt.show()
             
        return {
            "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "evs": evs,
            "plots": plots
        }
