"""
Functions for loading and preprocessing the customer churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

def download_dataset():
    """
    Download the Telco Customer Churn dataset from Kaggle or use a local copy.
    For this example, we'll use a direct download link to a public dataset.
    """
    import requests
    
    # URL for the Telco Customer Churn dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Local path to save the dataset
    local_path = RAW_DATA_DIR / "telco_customer_churn.csv"
    
    # Download the dataset if it doesn't exist
    if not local_path.exists():
        print(f"Downloading dataset to {local_path}...")
        response = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(response.content)
        print("Download complete!")
    else:
        print(f"Dataset already exists at {local_path}")
    
    return local_path

def load_raw_data(file_path=None):
    """
    Load the raw customer churn dataset.
    
    Args:
        file_path: Path to the dataset. If None, use the default path.
        
    Returns:
        pandas.DataFrame: The raw dataset.
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "telco_customer_churn.csv"
    
    # Check if file exists, if not, download it
    if not os.path.exists(file_path):
        file_path = download_dataset()
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    return df

def preprocess_data(df):
    """
    Preprocess the raw customer churn dataset.
    
    Args:
        df: pandas.DataFrame with the raw data.
        
    Returns:
        pandas.DataFrame: The preprocessed dataset.
    """
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert 'TotalCharges' to numeric, handling spaces
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df_processed['TotalCharges'].fillna(0, inplace=True)
    
    # Convert target variable to binary
    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    
    # Handle categorical variables
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col != 'customerID':  # Skip ID column
            df_processed[col] = df_processed[col].astype('category')
    
    print(f"Preprocessing complete. Dataset shape: {df_processed.shape}")
    
    return df_processed

def split_data(df):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        df: pandas.DataFrame with the preprocessed data.
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=VALIDATION_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save the processed datasets to disk.
    
    Args:
        X_train, X_val, X_test: Feature sets for training, validation, and test
        y_train, y_val, y_test: Target variables for training, validation, and test
    """
    # Create directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save the datasets
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROCESSED_DATA_DIR / "X_val.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
    
    print(f"Processed datasets saved to {PROCESSED_DATA_DIR}")

def main():
    """
    Main function to execute the data processing pipeline.
    """
    print("Starting data processing pipeline...")
    
    # Load raw data
    raw_data = load_raw_data()
    
    # Preprocess data
    processed_data = preprocess_data(raw_data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)
    
    # Save processed data
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("Data processing pipeline completed successfully!")

if __name__ == "__main__":
    main()