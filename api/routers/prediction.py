from fastapi import APIRouter, HTTPException
import pandas as pd
import os
import joblib
from typing import List, Dict, Any
from ..schemas import PredictionRequest

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)

MODELS_DIR = "trained_models"

@router.post("/{model_id}")
async def predict(model_id: str, request: PredictionRequest):
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found. Please train it first.")
        
        # Load Model
        model = joblib.load(model_path)
        
        # Create DataFrame from input data
        df = pd.DataFrame(request.data)
        
        # Make Prediction
        if hasattr(model, 'predict'):
            predictions = model.predict(df)
            return {"predictions": predictions.tolist()}
        else:
             raise HTTPException(status_code=400, detail="Loaded object is not a valid predictor.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
