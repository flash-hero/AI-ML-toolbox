from fastapi import APIRouter, HTTPException
import pandas as pd
import os
import joblib
import numpy as np
from ..schemas import TrainingRequest
from toolbox.data_collection import DataImporter # Potentially not needed if we read file directly
from toolbox.preprocessor import DataPreprocessor # Method_LogisticRegression_Classifier, Method_Linear_Regressor # Need to fix these imports, they are not in preprocessor

# Correct Imports from Toolbox structure
from toolbox.classificationModels.MLClassificationModels.KNNClassifier import Method_KNN_Classifier
from toolbox.classificationModels.MLClassificationModels.DecisionTreeClassifier import Method_DecisionTree_Classifier
from toolbox.classificationModels.MLClassificationModels.RandomForestClassifier import Method_RandomForest
from toolbox.classificationModels.MLClassificationModels.LDAClassifier import Method_LDA_Classifier
from toolbox.classificationModels.MLClassificationModels.NaiveBayesClassifier import Method_NaiveBayes_Classifier
from toolbox.classificationModels.MLClassificationModels.LogisticRegressionClassifier import Method_LogisticRegression_Classifier
from toolbox.classificationModels.MLClassificationModels.SVMClassifier import Method_SVM_Classifier
from toolbox.classificationModels.MLClassificationModels.QDAClassifier import Method_QDA_Classifier
from toolbox.classificationModels.MLClassificationModels.XGBoostClassifier import Method_XGBoost_Classifier
# LightGBM/CatBoost might have dep issues, include if installed
try:
    from toolbox.classificationModels.MLClassificationModels.LightGBMClassifier import Method_LightGBM_Classifier
    from toolbox.classificationModels.MLClassificationModels.CatBoostClassifier import Method_CatBoost_Classifier
except ImportError:
    pass
from toolbox.classificationModels.MLClassificationModels.AdaBoostClassifier import Method_AdaBoost_Classifier
from toolbox.classificationModels.MLClassificationModels.BaggingClassifier import Method_Bagging_Classifier
from toolbox.classificationModels.MLClassificationModels.ExtraTreesClassifier import Method_ExtraTrees_Classifier

from toolbox.regressionModels.MLRegressionModels.LinearRegressor import Method_Linear_Regressor
# ... Add other regression models same way ...

# RNN Models
from toolbox.classificationModels.RNNClassificationModels.LSTMClassifier import Method_LSTM_Classifier
from toolbox.classificationModels.RNNClassificationModels.GRUClassifier import Method_GRU_Classifier
# ... RNN Regressors ...

from toolbox.evaluationModels.evaluation_classification import ClassifierEvaluator
from toolbox.evaluationModels.evaluation_regressor import RegressionEvaluator
from toolbox.clusteringModels.kmeans import KMeansClustering

router = APIRouter(
    prefix="/train",
    tags=["training"]
)

PROCESSED_DIR = "processed_data"
MODELS_DIR = "trained_models"
os.makedirs(MODELS_DIR, exist_ok=True)

@router.post("/{task_type}/{model_name}")
async def train_model(task_type: str, model_name: str, request: TrainingRequest):
    try:
        # 1. Load Data
        file_path = os.path.join(PROCESSED_DIR, request.filename)
        if not os.path.exists(file_path):
             # Try upload dir
             file_path = os.path.join("uploaded_data", request.filename)
             if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Training data not found.")
        
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        # 2. Prepare Data (Split X, y)
        preprocessor = DataPreprocessor(df)
        
        X_train, X_test, y_train, y_test = None, None, None, None
        
        if request.task_type not in ["regression", "classification", "clustering"]:
             raise HTTPException(status_code=400, detail="Invalid task type")
             
        # Prepare plotting directory
        STATIC_DIR = "static/plots"
        model_id = f"{request.task_type}_{request.model_name}_{request.filename.replace('.', '_')}"
        model_plot_dir = os.path.join(STATIC_DIR, model_id)
        os.makedirs(model_plot_dir, exist_ok=True)
        
        # --- Model Selection Logic ---
        model_instance = None
        
        # Helper to instantiate model by name
        def get_model_instance(name, task, is_rnn=False):
            name = name.lower()
            if task == 'classification':
                if is_rnn:
                    if 'lstm' in name: return Method_LSTM_Classifier()
                    if 'gru' in name: return Method_GRU_Classifier()
                else:
                    if 'logistic' in name: return Method_LogisticRegression_Classifier()
                    if 'random_forest' in name or 'rf' in name: return Method_RandomForest()
                    if 'knn' in name: return Method_KNN_Classifier()
                    if 'decision_tree' in name: return Method_DecisionTree_Classifier()
                    if 'svm' in name: return Method_SVM_Classifier()
                    if 'naive_bayes' in name: return Method_NaiveBayes_Classifier()
                    if 'xgboost' in name: return Method_XGBoost_Classifier()
                    # Add others...
            elif task == 'regression':
                if is_rnn:
                    # if 'lstm' in name: return Method_LSTM_Regressor()
                    pass
                else:
                    if 'linear' in name: return Method_Linear_Regressor()
                    # Add others...
            return None

        model_instance = get_model_instance(request.model_name, request.task_type, request.is_time_series)
        
        if model_instance is None:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not supported for task '{request.task_type}'")

        # --- Data Splitting and Training ---
        results = {}
        
        if request.is_time_series:
            # Time Series / RNN Flow
            X_train, X_test, y_train, y_test = preprocessor.split_sequential_data(
                target_col=request.target_column,
                sequence_length=request.sequence_length, # Pass from request
                test_size=request.test_size
            )
            
            # RNN models usually have a different train method signature or we unify it
            # Checked LSTMClassifier: `train_lstm(X_train, y_train, X_test, y_test, n_iter=10)`
            # Checked RandomForest: `train_rf(X_train, y_train)`
            
            # We might need to handle specific method calls based on model type or try to refactor models to have common interface.
            # For now, let's check attributes.
            if hasattr(model_instance, 'train_lstm'):
                model_instance.train_lstm(X_train, y_train, X_test, y_test, n_iter=5) # Reduced iter for API speed
            elif hasattr(model_instance, 'train_gru'):
                model_instance.train_gru(X_train, y_train, X_test, y_test, n_iter=5)
            # Add Regression RNNs...
            
        else:
            # Standard ML Flow
            X_train, X_test, y_train, y_test = preprocessor.split_data(
                target_col=request.target_column,
                test_size=request.test_size,
                random_state=request.random_state
            )
            
            # Dynamic Training Call
            # Most ML models in toolbox have a specific train method like `train_rf`, `train_logistic`.
            # This is slightly inconsistent. I should probably standardise this in a refactor, 
            # but for now I will check for the specific method or valid pattern.
            
            method_name = f"train_{request.model_name.lower().replace('linear_regression', 'linear_regression').replace('logistic_regression', 'logistic').replace('random_forest', 'rf')}"
            
            # Fallback or specific mapping
            if hasattr(model_instance, method_name):
                 getattr(model_instance, method_name)(X_train, y_train)
            elif hasattr(model_instance, 'train'): # Some might have generic train
                 getattr(model_instance, 'train')(X_train, y_train)
            elif hasattr(model_instance, 'fit'): # Scikit-learn style
                 model_instance.fit(X_train, y_train)
            else:
                 # Try to brute-force find the train method? Or manual mapping.
                 if isinstance(model_instance, Method_Linear_Regressor):
                     model_instance.train_linear_regression(X_train, y_train)
                 elif isinstance(model_instance, Method_LogisticRegression_Classifier):
                     model_instance.train_logistic(X_train, y_train)
                 elif isinstance(model_instance, Method_RandomForest):
                     model_instance.train_rf(X_train, y_train)
                 elif isinstance(model_instance, Method_DecisionTree_Classifier):
                     model_instance.train_decision_tree(X_train, y_train)
                 # ... Add others ...

        # --- Evaluation ---
        # Predict
        y_pred = model_instance.predict(X_test)
        
        models_dir = "trained_models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{model_id}.pkl")
        
        # Save Model (handling different types)
        if hasattr(model_instance, 'model'):
             joblib.dump(model_instance.model, model_path)
        elif hasattr(model_instance, 'best_parameter'): # LSTM/RNN
             # It already saves inside the class, but let's see where.
             # It saves to 'best_lstm_classifier.h5'. We should move it or rename.
             # For now, let's rely on the class's saving if it does it, or try to save the object.
             # Keras models need special handling.
             if hasattr(model_instance.best_parameter, 'save'):
                 model_instance.best_parameter.save(model_path.replace('.pkl', '.h5'))
        else:
             joblib.dump(model_instance, model_path)

        if request.task_type == 'classification':
             evaluator = ClassifierEvaluator(y_test, y_pred)
             metrics = evaluator.evaluation_metrics(save_dir=model_plot_dir)
             results = {
                 "model_id": model_id,
                 "message": "Model trained successfully",
                 "metrics": metrics,
                 "plots": [f"/static/plots/{model_id}/confusion_matrix.png"] if metrics.get('confusion_matrix_path') else []
             }
        elif request.task_type == 'regression':
             evaluator = RegressionEvaluator(y_test, y_pred)
             metrics = evaluator.evaluation_metrics(save_dir=model_plot_dir)
             results = {
                 "model_id": model_id,
                 "message": "Model trained successfully",
                 "metrics": metrics,
                 "plots": [f"/static/plots/{model_id}/{os.path.basename(p)}" for p in metrics.get('plots', {}).values()]
             }
             
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
