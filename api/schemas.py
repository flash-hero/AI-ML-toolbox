from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class DatasetInfo(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    dtypes: Dict[str, str]

class ColumnPreview(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]

class PreprocessingRequest(BaseModel):
    filename: str
    target_column: Optional[str] = None
    drop_columns: Optional[List[str]] = None
    scale_method: Optional[str] = "standard" # standard, minmax
    encode_method: Optional[str] = "onehot" # onehot, label, target
    impute_strategy: Optional[str] = "mean" # mean, median, most_frequent
    
    # New fields for Time Series & Feature Selection
    is_time_series: Optional[bool] = False
    date_column: Optional[str] = None
    auto_select_features: Optional[bool] = False
    k_features: Optional[int] = 10

class TrainingRequest(BaseModel):
    filename: str
    task_type: str # regression, classification, clustering
    model_name: str
    target_column: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42
    
    # New fields for Time Series
    is_time_series: Optional[bool] = False
    sequence_length: Optional[int] = 10
    params: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]] # List of rows to predict
