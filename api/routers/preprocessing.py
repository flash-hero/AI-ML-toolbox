from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from ..schemas import PreprocessingRequest, DatasetInfo
from toolbox.preprocessor import DataPreprocessor

router = APIRouter(
    prefix="/preprocess",
    tags=["preprocessing"]
)

UPLOAD_DIR = "uploaded_data" 
# In a real app, this should be configurable or shared
PROCESSED_DIR = "processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

@router.post("/clean", response_model=DatasetInfo)
async def clean_data(request: PreprocessingRequest):
    try:
        file_path = os.path.join(UPLOAD_DIR, request.filename)
        if not os.path.exists(file_path):
             # Try looking in processed dir if it's a chained operation
             file_path = os.path.join(PROCESSED_DIR, request.filename)
             if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")

        # Load Data
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Initialize Preprocessor
        # Note: DataPreprocessor was refactored to take df in constructor
        preprocessor = DataPreprocessor(df)

        # Apply Cleaning
        # 1. Drop Columns if specified
        if request.drop_columns:
            preprocessor.data.drop(columns=request.drop_columns, inplace=True, errors='ignore')

        # 2. Impute Missing Values
        preprocessor.handle_missing_values(numeric_strategy=request.impute_strategy)
        
        # 3. Save processed file
        processed_filename = f"cleaned_{request.filename}"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)
        
        # Save as CSV for consistency in processing pipeline
        preprocessor.data.to_csv(processed_path, index=False)
        
        # Return info
        dtypes = {col: str(dtype) for col, dtype in preprocessor.data.dtypes.items()}
        
        return DatasetInfo(
            filename=processed_filename,
            rows=preprocessor.data.shape[0],
            columns=preprocessor.data.shape[1],
            column_names=list(preprocessor.data.columns),
            dtypes=dtypes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/encode", response_model=DatasetInfo)
async def encode_data(request: PreprocessingRequest):
    try:
        # Expecting file to be in PROCESSED_DIR or UPLOAD_DIR
        file_path = os.path.join(PROCESSED_DIR, request.filename)
        if not os.path.exists(file_path):
             file_path = os.path.join(UPLOAD_DIR, request.filename)
             if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        preprocessor = DataPreprocessor(df)
        
        # Trigger encoding
        preprocessor.encode_categorical_columns(method=request.encode_method, target_col=request.target_column) 
        
        encoded_filename = f"encoded_{request.filename}"
        encoded_path = os.path.join(PROCESSED_DIR, encoded_filename)
        preprocessor.data.to_csv(encoded_path, index=False)

        dtypes = {col: str(dtype) for col, dtype in preprocessor.data.dtypes.items()}
        
        return DatasetInfo(
            filename=encoded_filename,
            rows=preprocessor.data.shape[0],
            columns=preprocessor.data.shape[1],
            column_names=list(preprocessor.data.columns),
            dtypes=dtypes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scale", response_model=DatasetInfo)
async def scale_data(request: PreprocessingRequest):
    try:
        file_path = os.path.join(PROCESSED_DIR, request.filename)
        if not os.path.exists(file_path):
             file_path = os.path.join(UPLOAD_DIR, request.filename)
             if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        preprocessor = DataPreprocessor(df)
        
        # Trigger scaling
        preprocessor.normalize_numeric_columns(method=request.scale_method)

        scaled_filename = f"scaled_{request.filename}"
        scaled_path = os.path.join(PROCESSED_DIR, scaled_filename)
        preprocessor.data.to_csv(scaled_path, index=False)

        dtypes = {col: str(dtype) for col, dtype in preprocessor.data.dtypes.items()}
        
        return DatasetInfo(
            filename=scaled_filename,
            rows=preprocessor.data.shape[0],
            columns=preprocessor.data.shape[1],
            column_names=list(preprocessor.data.columns),
            dtypes=dtypes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process", response_model=DatasetInfo)
async def process_data(request: PreprocessingRequest):
    """
    Unified endpoint for all preprocessing steps:
    1. Cleaning (Imputation)
    2. Feature Selection (Optional)
    3. Scaling
    4. Encoding
    5. Time Series Preparation (Optional)
    """
    try:
        # Load Data (Check Processed first, then Upload)
        file_path = os.path.join(PROCESSED_DIR, request.filename)
        if not os.path.exists(file_path):
             file_path = os.path.join(UPLOAD_DIR, request.filename)
             if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")

        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        preprocessor = DataPreprocessor(df)
        
        # 1. Cleaning
        if request.drop_columns:
            preprocessor.data.drop(columns=request.drop_columns, inplace=True, errors='ignore')
            
        preprocessor.handle_missing_values(numeric_strategy=request.impute_strategy)
        
        # 2. Feature Selection (Optional)
        if request.auto_select_features:
            if not request.target_column:
                 raise HTTPException(status_code=400, detail="Target column required for feature selection.")
            
            # We need to temporarily encode/scale to run selection, but FeatureSelector handles some of this.
            # However, FeatureSelector expects X and y.
            target = preprocessor.data[request.target_column]
            features = preprocessor.data.drop(columns=[request.target_column])
            
            from toolbox.feature_selection import FeatureSelector
            selector = FeatureSelector(features, target)
            X_selected, _ = selector.select_features(method='auto', k=request.k_features)
            # select_features returns dataframe of features
            
            # Update data with selected features + target
            preprocessor.data = pd.concat([X_selected, target], axis=1)
            
        # 3. Scaling
        preprocessor.normalize_numeric_columns(method=request.scale_method)
        
        # 4. Encoding
        preprocessor.encode_categorical_columns(method=request.encode_method, target_col=request.target_column)

        # 5. Time Series Preparation (Optional)
        if request.is_time_series:
            if not request.date_column:
                 # Try to auto-detect
                 datetime_cols = [col for col in preprocessor.data.columns if pd.api.types.is_datetime64_any_dtype(preprocessor.data[col])]
                 if datetime_cols:
                     date_col = datetime_cols[0]
                 else:
                     # Try to convert text columns to date
                     preprocessor.convert_date_columns()
                     datetime_cols = [col for col in preprocessor.data.columns if pd.api.types.is_datetime64_any_dtype(preprocessor.data[col])]
                     if datetime_cols:
                         date_col = datetime_cols[0]
                     else:
                         raise HTTPException(status_code=400, detail="Time Series selected but no date column found. Please name one.")
            else:
                 date_col = request.date_column
            
            preprocessor.prepare_time_series_data(date_col=date_col)

        # Save processed data
        output_filename = f"processed_{request.filename}"
        output_path = os.path.join(PROCESSED_DIR, output_filename)
        preprocessor.data.to_csv(output_path, index=False)
        
        dtypes = {col: str(dtype) for col, dtype in preprocessor.data.dtypes.items()}

        return DatasetInfo(
            filename=output_filename,
            rows=preprocessor.data.shape[0],
            columns=preprocessor.data.shape[1],
            column_names=preprocessor.data.columns.tolist(),
            dtypes=dtypes
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
