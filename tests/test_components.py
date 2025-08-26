"""
Unit tests for the customer churn prediction components.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import preprocess_data
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer

class TestDataLoader(unittest.TestCase):
    """
    Tests for the data loader module.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a small test dataset
        self.test_data = pd.DataFrame({
            'customerID': ['1234-ABCD', '2345-BCDE', '3456-CDEF'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes'],
            'tenure': [12, 24, 36],
            'PhoneService': ['Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No internet service'],
            'TechSupport': ['No', 'No', 'No internet service'],
            'StreamingTV': ['No', 'Yes', 'No internet service'],
            'StreamingMovies': ['No', 'Yes', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
            'MonthlyCharges': [29.85, 56.95, 19.70],
            'TotalCharges': ['358.20', '1366.80', '709.20'],
            'Churn': ['Yes', 'No', 'No']
        })
    
    def test_preprocess_data(self):
        """
        Test the preprocess_data function.
        """
        # Preprocess the test data
        processed_data = preprocess_data(self.test_data)
        
        # Check that the output is a DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame)
        
        # Check that the target variable is binary
        self.assertTrue(set(processed_data['Churn']).issubset({0, 1}))
        
        # Check that TotalCharges is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_data['TotalCharges']))
        
        # Check that categorical columns are properly encoded
        for col in ['gender', 'Partner', 'Dependents', 'Contract']:
            self.assertTrue(pd.api.types.is_categorical_dtype(processed_data[col]))

class TestFeatureEngineering(unittest.TestCase):
    """
    Tests for the feature engineering module.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a small test dataset
        self.test_data = pd.DataFrame({
            'customerID': ['1234-ABCD', '2345-BCDE', '3456-CDEF'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes'],
            'tenure': [12, 24, 36],
            'PhoneService': ['Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No internet service'],
            'TechSupport': ['No', 'No', 'No internet service'],
            'StreamingTV': ['No', 'Yes', 'No internet service'],
            'StreamingMovies': ['No', 'Yes', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
            'MonthlyCharges': [29.85, 56.95, 19.70],
            'TotalCharges': [358.20, 1366.80, 709.20]
        })
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
    
    def test_create_interaction_features(self):
        """
        Test the create_interaction_features method.
        """
        # Create interaction features
        enhanced_data = self.feature_engineer.create_interaction_features(self.test_data)
        
        # Check that new features were created
        self.assertIn('tenure_by_MonthlyCharges', enhanced_data.columns)
        self.assertIn('avg_monthly_charges', enhanced_data.columns)
        self.assertIn('is_long_term', enhanced_data.columns)
        self.assertIn('is_high_value', enhanced_data.columns)
        
        # Check the values of the new features
        self.assertEqual(enhanced_data['tenure_by_MonthlyCharges'][0], self.test_data['tenure'][0] * self.test_data['MonthlyCharges'][0])
        self.assertEqual(enhanced_data['avg_monthly_charges'][0], self.test_data['TotalCharges'][0] / self.test_data['tenure'][0])
    
    def test_create_service_count_feature(self):
        """
        Test the create_service_count_feature method.
        """
        # Create service count feature
        enhanced_data = self.feature_engineer.create_service_count_feature(self.test_data)
        
        # Check that the new feature was created
        self.assertIn('service_count', enhanced_data.columns)
        
        # Check the values of the new feature
        # First customer has PhoneService=Yes, OnlineBackup=Yes, and all other services are No
        self.assertEqual(enhanced_data['service_count'][0], 2)
        
        # Second customer has PhoneService=Yes, MultipleLines=Yes, InternetService=Fiber optic,
        # DeviceProtection=Yes, StreamingTV=Yes, StreamingMovies=Yes
        self.assertEqual(enhanced_data['service_count'][1], 6)
        
        # Third customer has PhoneService=Yes, InternetService=No, and all internet-related services are "No internet service"
        self.assertEqual(enhanced_data['service_count'][2], 1)
    
    def test_fit_transform(self):
        """
        Test the fit_transform method.
        """
        # Apply fit_transform
        transformed_data = self.feature_engineer.fit_transform(self.test_data)
        
        # Check that the output is a numpy array or DataFrame
        self.assertTrue(isinstance(transformed_data, (np.ndarray, pd.DataFrame)))
        
        # Check that feature names are set
        self.assertTrue(hasattr(self.feature_engineer, 'feature_names'))
        self.assertTrue(len(self.feature_engineer.feature_names) > 0)

class TestModelTrainer(unittest.TestCase):
    """
    Tests for the model trainer module.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a small test dataset
        np.random.seed(42)
        n_samples = 100
        
        # Features
        X = np.random.rand(n_samples, 5)
        
        # Target (binary classification)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        # Split into train and test
        self.X_train = X[:80]
        self.y_train = y[:80]
        self.X_test = X[80:]
        self.y_test = y[80:]
    
    def test_model_initialization(self):
        """
        Test model initialization.
        """
        # Initialize models
        model_types = ['xgboost', 'random_forest', 'logistic_regression']
        
        for model_type in model_types:
            model_trainer = ModelTrainer(model_type=model_type)
            self.assertIsNotNone(model_trainer.model)
    
    def test_model_training_and_evaluation(self):
        """
        Test model training and evaluation.
        """
        # Initialize model trainer
        model_trainer = ModelTrainer(model_type='logistic_regression')
        
        # Train the model
        model_trainer.train(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model_trainer.predict(self.X_test)
        
        # Check that predictions have the right shape
        self.assertEqual(len(y_pred), len(self.y_test))
        
        # Evaluate the model
        metrics = model_trainer.evaluate(self.X_test, self.y_test)
        
        # Check that metrics were calculated
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('roc_auc', metrics)

if __name__ == '__main__':
    unittest.main()