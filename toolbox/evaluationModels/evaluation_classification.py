import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report

class ClassifierEvaluator:
    """
    This class evaluates the performance of a Classification model.
    It compares the 'True' answers with the 'Predicted' answers to calculate scores.
    """
    
    def __init__(self, y_true, y_pred, y_scores=None):
        """
        Initializes the evaluator.
        
        Parameters:
        - y_true: The actual correct labels (ground truth).
        - y_pred: The labels predicted by the model.
        - y_scores: (Optional) The raw probability scores output by the model.
        """
        # Ensure we have the same number of answers and predictions
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of y_true and y_pred must be equal.")
            
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_scores = y_scores

        # Print unique classes found in the data (debugging info)
        print(f"y_true classes: {np.unique(self.y_true)}")
        print(f"y_pred classes: {np.unique(self.y_pred)}")
        
        self.classification_type = self.determine_classification_type()

    def determine_classification_type(self):
        """
        Determines if this is a Binary (2 classes) or Multiclass (>2 classes) problem.
        """
        unique_classes = np.unique(self.y_true)
        if len(unique_classes) == 2:
            return 'binary'
        else:
            return 'multiclass'

    def calculate_accuracy(self):
        """
        Calculates Accuracy: The percentage of correct predictions.
        (Correct / Total)
        """
        return accuracy_score(self.y_true, self.y_pred)

    def calculate_precision(self):
        """
        Calculates Precision: Out of all positive predictions, how many were actually positive?
        (True Positives / Predicted Positives)
        """
        return precision_score(self.y_true, self.y_pred, average='weighted')

    def calculate_recall(self):
        """
        Calculates Recall: Out of all actual positives, how many did we find?
        (True Positives / Actual Positives)
        """
        return recall_score(self.y_true, self.y_pred, average='weighted')

    def calculate_f1_score(self):
        """
        Calculates F1 Score: A balance between Precision and Recall.
        Useful when classes are uneven.
        """
        return f1_score(self.y_true, self.y_pred, average='weighted')

    def get_confusion_matrix(self, save_path=None):
        """
        Generates and displays a Confusion Matrix.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 
    
    def evaluation_metrics(self, save_dir=None):
        """
        Calculates all metrics and prints a summary report.
        """
        accuracy = self.calculate_accuracy()
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1_score()
        
        # Detailed report by class
        class_report = classification_report(self.y_true, self.y_pred)

        print("\n----- Classification Evaluation Report -----")
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.2%}")
        print("\nDetailed Report:")
        print(class_report)
        
        import os
        cm_path = os.path.join(save_dir, "confusion_matrix.png") if save_dir else None
        self.get_confusion_matrix(save_path=cm_path)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": class_report,
            "confusion_matrix_path": cm_path if save_dir else None
        }
