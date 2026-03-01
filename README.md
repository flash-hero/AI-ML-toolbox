# 🧠 AI/ML Toolbox — End-to-End Machine Learning Platform

A full-stack machine learning platform that automates the entire data science lifecycle — from data ingestion and preprocessing to model training, evaluation, and prediction. Supports **Classification**, **Regression**, and **Clustering** tasks with **53 implemented algorithms**, including deep learning.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Models Implemented (53)](#-models-implemented-53)
- [Features](#-features)
- [API Endpoints](#-api-endpoints)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Testing](#-testing)
- [Contributing](#-contributing)

---

## 🔍 Overview

This platform provides:

- **A REST API** (FastAPI) for programmatic access to the full ML pipeline
- **A Web UI** (Single Page Application) for interactive, no-code usage
- **A CLI** (interactive terminal menu) for power users
- **53 ML/DL algorithms** with automated hyperparameter tuning
- **Statistical feature selection** and comprehensive **EDA visualizations**
- **Model persistence** and **inference on new data**

---

## 🏗 Architecture

```
┌─────────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────────────┐
│   Frontend (Web UI)     │     │     REST API (FastAPI)    │     │       ML Engine (Toolbox)        │
│                         │     │                          │     │                                 │
│  HTML5 / CSS3 / JS SPA  │────▶│  /data                   │────▶│  ├── 53 ML/DL Models            │
│  Drag & Drop Upload     │     │  /preprocess             │     │  ├── Data Preprocessor          │
│  Interactive Controls    │     │  /train                  │     │  ├── Feature Selector           │
│  Real-time Results       │◀────│  /predict                │◀────│  ├── Evaluation Modules         │
│                         │     │  /visualization          │     │  └── Data Visualizer            │
└─────────────────────────┘     └──────────────────────────┘     └─────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | Python, FastAPI, Pydantic, Uvicorn |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | TensorFlow, Keras, Keras Tuner (LSTM, GRU) |
| **Data Processing** | Pandas, NumPy, SciPy, Category Encoders |
| **Visualization** | Matplotlib, Seaborn |
| **Statistical Analysis** | SciPy (ANOVA, Chi-Square, T-test, Spearman, Shapiro-Wilk, Levene's) |
| **Clustering (specialized)** | HDBSCAN, scikit-fuzzy, scikit-learn-extra, kmodes, MiniSOM, NetworkX |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla SPA) |
| **Model Serialization** | Joblib (.pkl), Keras (.h5) |
| **Testing** | FastAPI TestClient (E2E integration tests) |

---

## 🤖 Models Implemented (53)

### ML Classification (14)
| # | Model | Hyperparameter Tuning |
|---|-------|-----------------------|
| 1 | Logistic Regression | RandomizedSearchCV |
| 2 | K-Nearest Neighbors (KNN) | RandomizedSearchCV |
| 3 | Decision Tree | RandomizedSearchCV |
| 4 | Random Forest | RandomizedSearchCV + Cross-Validation |
| 5 | Support Vector Machine (SVM) | RandomizedSearchCV |
| 6 | Naive Bayes (Gaussian) | RandomizedSearchCV |
| 7 | Linear Discriminant Analysis (LDA) | RandomizedSearchCV |
| 8 | Quadratic Discriminant Analysis (QDA) | RandomizedSearchCV |
| 9 | XGBoost | RandomizedSearchCV |
| 10 | LightGBM | RandomizedSearchCV |
| 11 | CatBoost | RandomizedSearchCV |
| 12 | AdaBoost | RandomizedSearchCV |
| 13 | Bagging Classifier | RandomizedSearchCV |
| 14 | Extra Trees | RandomizedSearchCV |

### ML Regression (14)
| # | Model | Hyperparameter Tuning |
|---|-------|-----------------------|
| 1 | Linear Regression | RandomizedSearchCV |
| 2 | KNN Regressor | RandomizedSearchCV |
| 3 | Decision Tree Regressor | RandomizedSearchCV |
| 4 | Random Forest Regressor | RandomizedSearchCV |
| 5 | Support Vector Regressor (SVR) | RandomizedSearchCV |
| 6 | Bayesian Linear Regression | RandomizedSearchCV |
| 7 | Bayesian Ridge Regression | RandomizedSearchCV |
| 8 | Gaussian Process Regressor | RandomizedSearchCV |
| 9 | Gradient Boosting Regressor | RandomizedSearchCV |
| 10 | AdaBoost Regressor | RandomizedSearchCV |
| 11 | XGBoost Regressor | RandomizedSearchCV |
| 12 | CatBoost Regressor | RandomizedSearchCV |
| 13 | Extra Trees Regressor | RandomizedSearchCV |
| 14 | LightGBM Regressor | RandomizedSearchCV |

### Deep Learning (4)
| # | Model | Architecture | Tuning |
|---|-------|-------------|--------|
| 1 | LSTM Classifier | Stacked LSTM (1–3 layers, 32–128 units) | Keras Tuner RandomSearch |
| 2 | GRU Classifier | Stacked GRU + Dropout | Keras Tuner RandomSearch |
| 3 | LSTM Regressor | Stacked LSTM + Dense | Keras Tuner RandomSearch |
| 4 | GRU Regressor | Stacked GRU + Dropout + Dense | Keras Tuner RandomSearch |

### Clustering (21)
| # | Model | Type |
|---|-------|------|
| 1 | K-Means | Centroid-based |
| 2 | Mini-Batch K-Means | Centroid-based (scalable) |
| 3 | K-Medoids | Centroid-based (robust to outliers) |
| 4 | K-Prototypes | Mixed data (numeric + categorical) |
| 5 | K-Modes | Categorical data |
| 6 | CLARANS | Medoid-based |
| 7 | C-Means | Fuzzy/soft clustering |
| 8 | Fuzzy C-Means | Fuzzy/soft clustering |
| 9 | COP-KMeans | Constrained clustering |
| 10 | DBSCAN | Density-based |
| 11 | HDBSCAN | Density-based (hierarchical) |
| 12 | Mean Shift | Density-based |
| 13 | OPTICS | Density order-based |
| 14 | Agglomerative Hierarchical | Hierarchical (bottom-up) |
| 15 | Divisive Hierarchical | Hierarchical (top-down) |
| 16 | BIRCH | Incremental hierarchical |
| 17 | Agglomerative-BIRCH | Hybrid (BIRCH + Agglomerative) |
| 18 | Spectral Clustering | Graph-based |
| 19 | Affinity Propagation | Message-passing |
| 20 | Density Peaks | Density peak detection |
| 21 | Self-Organizing Maps (SOM) | Neural network-based |

---

## ✨ Features

### Data Ingestion
- File upload via **drag-and-drop** or file dialog (CSV, Excel, TXT)
- Automatic delimiter detection (comma, semicolon, tab, pipe)
- Data preview with configurable row count

### Preprocessing Pipeline
- **Missing value imputation** — mean, median, most frequent
- **Feature scaling** — StandardScaler, MinMaxScaler
- **Categorical encoding** — OneHot, Label, Target Encoding
- **Column management** — selective feature dropping
- **Time series handling** — datetime detection, frequency inference, missing date gap filling, date decomposition (Year, Month, Day, DayOfWeek, Hour)
- **Unified endpoint** — single API call chains all preprocessing steps

### Statistical Feature Selection
- Chi-Square test (categorical → categorical)
- Mutual Information (classification & regression)
- ANOVA / Kruskal-Wallis (numerical → categorical)
- Spearman correlation (numerical → numerical)
- Automatic normality testing (Shapiro-Wilk, Levene's) to choose parametric vs. non-parametric tests
- Top-K feature selection with configurable K

### Exploratory Data Analysis (EDA)
- Descriptive statistics (quantitative + qualitative)
- Boxplots for outlier detection
- Pair plots for feature relationships
- Pie charts for categorical distributions
- Correlation heatmaps
- Statistical hypothesis testing (T-test, Chi-Square)
- All plots exportable as PNG

### Hyperparameter Tuning
- **RandomizedSearchCV** — all 28 ML models
- **Keras Tuner RandomSearch** — all 4 DL models
- **ParameterGrid + Silhouette Score** — DBSCAN, HDBSCAN, K-Means

### Model Evaluation
- **Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix (heatmap), Classification Report
- **Regression:** MSE, RMSE, MAE, R², Explained Variance, Prediction vs. Actual plot, Error Distribution histogram
- **Clustering:** Silhouette Score, per-cluster descriptive statistics, ANOVA/Chi-Square feature importance, boxplots/violin plots per cluster

### Model Persistence & Prediction
- ML models saved as `.pkl` (Joblib)
- DL models saved as `.h5` (Keras)
- Prediction endpoint loads any trained model and returns predictions on new data

---

## 🔌 API Endpoints

| Router | Method | Endpoint | Description |
|--------|--------|----------|-------------|
| **Data** | `POST` | `/data/upload` | Upload a CSV or Excel file |
| | `GET` | `/data/preview/{filename}` | Preview first N rows |
| **Preprocessing** | `POST` | `/preprocess/clean` | Impute missing values, drop columns |
| | `POST` | `/preprocess/scale` | Scale numeric features |
| | `POST` | `/preprocess/encode` | Encode categorical variables |
| | `POST` | `/preprocess/process` | Unified pipeline (all steps) |
| **Training** | `POST` | `/train/{task_type}/{model_name}` | Train a model with evaluation |
| **Prediction** | `POST` | `/predict/{model_id}` | Run inference on new data |
| **Visualization** | `GET` | `/visualization/generate/{filename}` | Generate EDA plots |
| | `GET` | `/visualization/list/{filename}` | List available plots |
| **System** | `GET` | `/` | Root (status check) |
| | `GET` | `/health` | Health check |

Full interactive docs available at `/docs` (Swagger UI) after starting the server.

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-ml-toolbox.git
cd ai-ml-toolbox

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r toolbox/requirements.txt
pip install fastapi uvicorn python-multipart joblib

# 4. Start the API server
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be running at `http://127.0.0.1:8000` with Swagger docs at `http://127.0.0.1:8000/docs`.

### Using the Web UI

Open `ui/index.html` in your browser (or serve it via a local HTTP server) and it will connect to the API at `http://127.0.0.1:8000`.

### Using the CLI

```bash
cd toolbox
python menu.py
```

---

## 💡 Usage

### Via Web UI
1. **Upload** your dataset (CSV/Excel) using drag-and-drop
2. **Preprocess** — select imputation strategy, scaling method, encoding method
3. **Train** — choose task type (Classification/Regression/Clustering), select a model
4. **Evaluate** — view metrics and visualizations
5. **Predict** — submit new data for inference

### Via API (Python Example)

```python
import requests

BASE = "http://127.0.0.1:8000"

# Upload data
files = {"file": open("my_dataset.csv", "rb")}
r = requests.post(f"{BASE}/data/upload", files=files)
filename = r.json()["filename"]

# Preprocess
r = requests.post(f"{BASE}/preprocess/process", json={
    "filename": filename,
    "impute_strategy": "mean",
    "scale_method": "standard",
    "encode_method": "label"
})
processed = r.json()["filename"]

# Train
r = requests.post(f"{BASE}/train/classification/random_forest", json={
    "filename": processed,
    "target_column": "target",
    "task_type": "classification",
    "model_name": "random_forest"
})
model_id = r.json()["model_id"]
print("Metrics:", r.json()["metrics"])

# Predict
r = requests.post(f"{BASE}/predict/{model_id}", json={
    "model_id": model_id,
    "data": [{"feature1": 5.1, "feature2": 3.5}]
})
print("Prediction:", r.json()["predictions"])
```

---

## 📁 Project Structure

```
├── api/                            # FastAPI backend
│   ├── main.py                     # App entry point, CORS, router registration
│   ├── schemas.py                  # Pydantic request/response models
│   └── routers/
│       ├── data.py                 # Upload & preview endpoints
│       ├── preprocessing.py        # Clean, encode, scale, unified pipeline
│       ├── training.py             # Model training & evaluation
│       ├── prediction.py           # Model inference
│       └── visualization.py        # EDA plot generation
│
├── toolbox/                        # Core ML engine
│   ├── data_collection.py          # File import with auto-delimiter detection
│   ├── preprocessor.py             # Full preprocessing pipeline
│   ├── vis_analyse.py              # EDA & statistical visualizations
│   ├── feature_selection.py        # Statistical feature selection
│   ├── classificationModels/
│   │   ├── MLClassificationModels/ # 14 ML classifiers + orchestrator
│   │   └── RNNClassificationModels/# LSTM & GRU classifiers + orchestrator
│   ├── regressionModels/
│   │   ├── MLRegressionModels/     # 14 ML regressors + orchestrator
│   │   └── RNNRegressionModels/    # LSTM & GRU regressors + orchestrator
│   ├── clusteringModels/           # 21 clustering algorithms + analyzer
│   ├── evaluationModels/
│   │   ├── evaluation_classification.py
│   │   └── evaluation_regressor.py
│   └── requirements.txt
│
├── ui/                             # Frontend SPA
│   ├── index.html                  # Main HTML page
│   ├── css/
│   │   ├── style.css               # Layout & theme (CSS variables)
│   │   └── components.css          # UI components
│   └── js/
│       ├── api.js                  # API client (Fetch)
│       ├── ui.js                   # Page renderers & DOM management
│       └── app.js                  # App controller & state management
│
├── tests/
│   └── test_api.py                 # E2E integration tests
│
├── trained_models/                 # Persisted models (.pkl, .h5)
├── processed_data/                 # Preprocessed datasets
├── uploaded_data/                  # Raw uploaded files
├── static/plots/                   # Generated visualization images
├── menu.py                         # CLI entry point
└── README.md
```

---

## 🧪 Testing

```bash
# Run E2E integration tests
python tests/test_api.py
```

The test suite covers the complete pipeline: **Upload → Preview → Preprocess → Train → Predict**.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -m 'Add new model'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.
