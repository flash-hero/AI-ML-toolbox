from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import os
import shutil
from typing import List
from ..schemas import DatasetInfo, ColumnPreview

router = APIRouter(
    prefix="/data",
    tags=["data"]
)

# Directory to store uploaded files temporarily
UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DatasetInfo)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read file to get info
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_location)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_location)
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload CSV or Excel.")

        # Convert dtypes to string representation
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return DatasetInfo(
            filename=file.filename,
            rows=df.shape[0],
            columns=df.shape[1],
            column_names=list(df.columns),
            dtypes=dtypes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preview/{filename}", response_model=ColumnPreview)
async def preview_data(filename: str, rows: int = 5):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        
        preview_df = df.head(rows)
        # Handle nan values for JSON serialization
        preview_df = preview_df.fillna("NaN")
        
        return ColumnPreview(
            columns=list(preview_df.columns),
            data=preview_df.to_dict(orient="records")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
