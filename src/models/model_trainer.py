"""
Model training and evaluation for customer churn prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import joblib
import sys
import os
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, MODEL_PARAMS, RANDOM_STATE

class ModelTrainer:
    """
    Class for training and evaluating machine learning models for churn prediction.
    """
    
    def __init__(self, model_type='xgboost', params=None):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('xgboost', 'random_forest', or 'logistic_regression')
            params: Dictionary of model parameters (optional)
        """
        self.model_type = model_type
        self.model = None
        
        # Use default parameters if none provided
        if params is None:
            if model_type in MODEL_PARAMS:
                self.params = MODEL_PARAMS[model_type]
            else:
                self.params = {}
        else:
            self.params = params
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the model based on the specified type.
        """
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**self.params)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.params)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model on the training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            self: The trained model
        """
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            numpy.ndarray: Predicted classes
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            numpy.ndarray: Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y_true):
        """
        Evaluate the model performance.
        
        Args:
            X: Features to evaluate on
            y_true: True target values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'average_precision': average_precision_score(y_true, y_prob)
        }
        
        # Print metrics
        print(f"\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return metrics
    
    def plot_confusion_matrix(self, X, y_true, figsize=(10, 8)):
        """
        Plot the confusion matrix.
        
        Args:
            X: Features to evaluate on
            y_true: True target values
            figsize: Figure size (width, height) in inches
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {self.model_type.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        os.makedirs(f"{MODELS_DIR}/plots", exist_ok=True)
        plt.savefig(f"{MODELS_DIR}/plots/confusion_matrix_{self.model_type}.png")
        plt.close()
    
    def plot_roc_curve(self, X, y_true, figsize=(10, 8)):
        """
        Plot the ROC curve.
        
        Args:
            X: Features to evaluate on
            y_true: True target values
            figsize: Figure size (width, height) in inches
        """
        # Make probability predictions
        y_prob = self.predict_proba(X)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # Plot ROC curve
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_type.capitalize()}')
        plt.legend(loc='lower right')
        
        # Save the plot
        os.makedirs(f"{MODELS_DIR}/plots", exist_ok=True)
        plt.savefig(f"{MODELS_DIR}/plots/roc_curve_{self.model_type}.png")
        plt.close()
    
    def plot_precision_recall_curve(self, X, y_true, figsize=(10, 8)):
        """
        Plot the precision-recall curve.
        
        Args:
            X: Features to evaluate on
            y_true: True target values
            figsize: Figure size (width, height) in inches
        """
        # Make probability predictions
        y_prob = self.predict_proba(X)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Plot precision-recall curve
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_type.capitalize()}')
        plt.legend(loc='upper right')
        
        # Save the plot
        os.makedirs(f"{MODELS_DIR}/plots", exist_ok=True)
        plt.savefig(f"{MODELS_DIR}/plots/pr_curve_{self.model_type}.png")
        plt.close()
    
    def plot_feature_importance(self, feature_names, top_n=20, figsize=(12, 10)):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            figsize: Figure size (width, height) in inches
        """
        # Check if the model supports feature importance
        if not hasattr(self.model, 'feature_importances_'):
            print(f"Model {self.model_type} does not support feature importance.")
            return
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Create DataFrame for plotting
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Select top N features
        top_features = feature_importance.head(top_n)
        
        # Plot feature importance
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance - {self.model_type.capitalize()}')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(f"{MODELS_DIR}/plots", exist_ok=True)
        plt.savefig(f"{MODELS_DIR}/plots/feature_importance_{self.model_type}.png")
        plt.close()
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, param_grid, cv=5, n_iter=20, scoring='roc_auc'):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            param_grid: Dictionary of hyperparameter grid
            cv: Number of cross-validation folds
            n_iter: Number of parameter settings sampled
            scoring: Scoring metric
            
        Returns:
            self: The model with optimized hyperparameters
        """
        print(f"Performing hyperparameter tuning for {self.model_type}...")
        
        # Initialize the model for tuning
        if self.model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=RANDOM_STATE)
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(random_state=RANDOM_STATE)
        elif self.model_type == 'logistic_regression':
            model = LogisticRegression(random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train)
        
        # Print best parameters
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best {scoring} score: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.params = random_search.best_params_
        self._initialize_model()
        
        # Train model with best parameters
        self.train(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = self.evaluate(X_val, y_val)
        print(f"Validation {scoring}: {val_metrics.get(scoring, 'N/A'):.4f}")
        
        return self
    
    def save_model(self, filename=None):
        """
        Save the trained model to disk.
        
        Args:
            filename: Name of the file to save the model (optional)
            
        Returns:
            str: Path to the saved model
        """
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_{timestamp}.joblib"
        
        # Ensure filename has .joblib extension
        if not filename.endswith('.joblib'):
            filename += '.joblib'
        
        # Save the model
        model_path = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            ModelTrainer: Instance with the loaded model
        """
        # Load the model
        model = joblib.load(model_path)
        
        # Determine model type
        if isinstance(model, xgb.XGBClassifier):
            model_type = 'xgboost'
        elif isinstance(model, RandomForestClassifier):
            model_type = 'random_forest'
        elif isinstance(model, LogisticRegression):
            model_type = 'logistic_regression'
        else:
            model_type = 'unknown'
        
        # Create instance
        instance = cls(model_type=model_type)
        instance.model = model
        
        print(f"Model loaded from {model_path}")
        
        return instance

def main():
    """
    Main function to demonstrate model training and evaluation.
    """
    from src.data.data_loader import load_raw_data, preprocess_data, split_data
    from src.features.feature_engineering import FeatureEngineer
    
    print("Loading and preprocessing data...")
    raw_data = load_raw_data()
    processed_data = preprocess_data(raw_data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)
    
    # Apply feature engineering
    print("Applying feature engineering...")
    feature_engineer = FeatureEngineer()
    
    # Create additional features
    X_train = feature_engineer.create_interaction_features(X_train)
    X_train = feature_engineer.create_service_count_feature(X_train)
    
    X_val = feature_engineer.create_interaction_features(X_val)
    X_val = feature_engineer.create_service_count_feature(X_val)
    
    X_test = feature_engineer.create_interaction_features(X_test)
    X_test = feature_engineer.create_service_count_feature(X_test)
    
    # Transform features
    X_train_transformed = feature_engineer.fit_transform(X_train)
    X_val_transformed = feature_engineer.transform(X_val)
    X_test_transformed = feature_engineer.transform(X_test)
    
    # Train and evaluate model
    print("Training and evaluating model...")
    model_trainer = ModelTrainer(model_type='xgboost')
    model_trainer.train(X_train_transformed, y_train)
    
    # Evaluate on validation set
    print("\nValidation set evaluation:")
    model_trainer.evaluate(X_val_transformed, y_val)
    
    # Plot evaluation metrics
    model_trainer.plot_confusion_matrix(X_val_transformed, y_val)
    model_trainer.plot_roc_curve(X_val_transformed, y_val)
    model_trainer.plot_precision_recall_curve(X_val_transformed, y_val)
    
    # Get feature names if available
    feature_names = None
    if hasattr(feature_engineer, 'feature_names'):
        feature_names = feature_engineer.feature_names
        model_trainer.plot_feature_importance(feature_names)
    
    # Save the model
    model_trainer.save_model()
    
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main()