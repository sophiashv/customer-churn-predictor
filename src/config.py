"""
Configuration settings for the Customer Churn Prediction project.
"""

import os
from pathlib import Path

# Define project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data processing settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # percentage of training data

# Model training settings
TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]
CATEGORICAL_COLUMNS = [
    "gender", 
    "Partner", 
    "Dependents", 
    "PhoneService", 
    "MultipleLines", 
    "InternetService",
    "OnlineSecurity", 
    "OnlineBackup", 
    "DeviceProtection", 
    "TechSupport", 
    "StreamingTV",
    "StreamingMovies", 
    "Contract", 
    "PaperlessBilling", 
    "PaymentMethod"
]
NUMERICAL_COLUMNS = [
    "tenure", 
    "MonthlyCharges", 
    "TotalCharges"
]

# Model hyperparameters
# These are default values that can be overridden during hyperparameter tuning
MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE
    },
    "logistic_regression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "random_state": RANDOM_STATE
    }
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
MODEL_FILE = "churn_model.joblib"