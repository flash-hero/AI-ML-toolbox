import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the parent directory to sys.path to allow importing 'toolbox'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the toolbox directory to sys.path to allow imports within toolbox modules (e.g. from evaluationModels)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'toolbox'))

app = FastAPI(
    title="Toolbox API",
    description="API for the Machine Learning Toolbox (Regression, Classification, Clustering)",
    version="1.0.0"
)

from api.routers import data, preprocessing, training, prediction, visualization
from fastapi.staticfiles import StaticFiles
import os

# Create static dir if not exists
os.makedirs("static", exist_ok=True)

app.include_router(data.router)
app.include_router(preprocessing.router)
app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(visualization.router)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Origins for CORS (allow all for development)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Toolbox API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
