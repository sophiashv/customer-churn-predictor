"""
Feature engineering for the customer churn prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, ID_COLUMNS

class FeatureEngineer:
    """
    Class for feature engineering operations on the customer churn dataset.
    """
    
    def __init__(self):
        """
        Initialize the feature engineering pipeline.
        """
        # Define preprocessing for numerical columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, NUMERICAL_COLUMNS),
                ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
            ],
            remainder='drop'  # Drop columns not specified in transformers
        )
        
        self.feature_names = None
    
    def fit_transform(self, X, y=None):
        """
        Fit the feature engineering pipeline and transform the data.
        
        Args:
            X: pandas.DataFrame with input features
            y: pandas.Series with target variable (optional)
            
        Returns:
            numpy.ndarray: Transformed features
        """
        # Extract ID column if present
        id_values = None
        if any(col in X.columns for col in ID_COLUMNS):
            id_values = X[ID_COLUMNS].copy()
            
        # Fit and transform the data
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        self._set_feature_names()
        
        # Convert to DataFrame if ID column was present
        if id_values is not None:
            X_transformed_df = pd.DataFrame(
                X_transformed, 
                columns=self.feature_names
            )
            # Add back the ID column
            for id_col in ID_COLUMNS:
                if id_col in X.columns:
                    X_transformed_df[id_col] = id_values[id_col].values
            return X_transformed_df
        
        return X_transformed
    
    def transform(self, X):
        """
        Transform the data using the fitted feature engineering pipeline.
        
        Args:
            X: pandas.DataFrame with input features
            
        Returns:
            numpy.ndarray: Transformed features
        """
        # Extract ID column if present
        id_values = None
        if any(col in X.columns for col in ID_COLUMNS):
            id_values = X[ID_COLUMNS].copy()
            
        # Transform the data
        X_transformed = self.preprocessor.transform(X)
        
        # Convert to DataFrame if ID column was present
        if id_values is not None:
            X_transformed_df = pd.DataFrame(
                X_transformed, 
                columns=self.feature_names
            )
            # Add back the ID column
            for id_col in ID_COLUMNS:
                if id_col in X.columns:
                    X_transformed_df[id_col] = id_values[id_col].values
            return X_transformed_df
        
        return X_transformed
    
    def _set_feature_names(self):
        """
        Extract feature names from the preprocessing pipeline.
        """
        # Get feature names for numerical columns
        numerical_features = NUMERICAL_COLUMNS
        
        # Get feature names for one-hot encoded categorical columns
        categorical_features = []
        
        # Get the OneHotEncoder from the pipeline
        ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        
        # Get the feature names from the OneHotEncoder
        if hasattr(ohe, 'get_feature_names_out'):
            categorical_features = ohe.get_feature_names_out(CATEGORICAL_COLUMNS)
        else:
            # For older scikit-learn versions
            categorical_features = [f"{col}_{cat}" for i, col in enumerate(CATEGORICAL_COLUMNS) 
                                   for cat in ohe.categories_[i]]
        
        # Combine all feature names
        self.feature_names = list(numerical_features) + list(categorical_features)
    
    def create_interaction_features(self, X):
        """
        Create interaction features between selected columns.
        
        Args:
            X: pandas.DataFrame with input features
            
        Returns:
            pandas.DataFrame: DataFrame with added interaction features
        """
        X_new = X.copy()
        
        # Create interaction between tenure and MonthlyCharges
        if 'tenure' in X.columns and 'MonthlyCharges' in X.columns:
            X_new['tenure_by_MonthlyCharges'] = X['tenure'] * X['MonthlyCharges']
        
        # Create average monthly charge
        if 'TotalCharges' in X.columns and 'tenure' in X.columns:
            # Avoid division by zero
            X_new['avg_monthly_charges'] = X['TotalCharges'] / X['tenure'].replace(0, 1)
        
        # Create binary feature for long-term customers
        if 'tenure' in X.columns:
            X_new['is_long_term'] = (X['tenure'] > 24).astype(int)
        
        # Create binary feature for high-value customers
        if 'MonthlyCharges' in X.columns:
            X_new['is_high_value'] = (X['MonthlyCharges'] > X['MonthlyCharges'].median()).astype(int)
        
        return X_new
    
    def create_service_count_feature(self, X):
        """
        Create a feature counting the number of services each customer has.
        
        Args:
            X: pandas.DataFrame with input features
            
        Returns:
            pandas.DataFrame: DataFrame with added service count feature
        """
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        X_new = X.copy()
        
        # Count services that are not 'No' or 'No internet service'
        service_count = 0
        for col in service_columns:
            if col in X.columns:
                # Convert to string to handle categorical dtype
                service_count += ((X[col].astype(str) != 'No') & 
                                 (X[col].astype(str) != 'No internet service')).astype(int)
        
        X_new['service_count'] = service_count
        
        return X_new

def main():
    """
    Main function to demonstrate feature engineering.
    """
    from src.data.data_loader import load_raw_data, preprocess_data
    
    print("Loading and preprocessing data...")
    raw_data = load_raw_data()
    processed_data = preprocess_data(raw_data)
    
    print("Applying feature engineering...")
    feature_engineer = FeatureEngineer()
    
    # Create interaction features
    enhanced_data = feature_engineer.create_interaction_features(processed_data)
    
    # Create service count feature
    enhanced_data = feature_engineer.create_service_count_feature(enhanced_data)
    
    print(f"Original data shape: {processed_data.shape}")
    print(f"Enhanced data shape: {enhanced_data.shape}")
    
    # Show new features
    new_features = [col for col in enhanced_data.columns if col not in processed_data.columns]
    print(f"New features created: {new_features}")
    
    # Apply the feature engineering pipeline
    transformed_data = feature_engineer.fit_transform(enhanced_data)
    print(f"Transformed data shape: {transformed_data.shape}")
    
    if hasattr(transformed_data, 'columns'):
        print(f"Transformed feature names: {transformed_data.columns.tolist()[:5]}... (showing first 5)")
    else:
        print(f"Number of transformed features: {transformed_data.shape[1]}")

if __name__ == "__main__":
    main()