"""
Main script to run the entire customer churn prediction pipeline.
"""

import os
import sys
import argparse
import logging
import joblib
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import download_dataset, load_raw_data, preprocess_data, split_data, save_processed_data
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.config import MODELS_DIR, RANDOM_STATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the customer churn prediction pipeline.')
    parser.add_argument('--download', action='store_true', help='Download the dataset')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                        choices=['xgboost', 'random_forest', 'logistic_regression'],
                        help='Type of model to train')
    parser.add_argument('--use-smote', action='store_true', help='Use SMOTE for handling class imbalance')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Perform hyperparameter tuning')
    
    return parser.parse_args()

def run_data_processing():
    """
    Run the data processing pipeline.
    """
    logger.info("Starting data processing pipeline...")
    
    # Download dataset
    file_path = download_dataset()
    
    # Load raw data
    raw_data = load_raw_data(file_path)
    
    # Preprocess data
    processed_data = preprocess_data(raw_data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)
    
    # Save processed data
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    logger.info("Data processing pipeline completed successfully!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_feature_engineering(X_train, X_val, X_test):
    """
    Run the feature engineering pipeline.
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create additional features
    logger.info("Creating additional features...")
    X_train = feature_engineer.create_interaction_features(X_train)
    X_train = feature_engineer.create_service_count_feature(X_train)
    
    X_val = feature_engineer.create_interaction_features(X_val)
    X_val = feature_engineer.create_service_count_feature(X_val)
    
    X_test = feature_engineer.create_interaction_features(X_test)
    X_test = feature_engineer.create_service_count_feature(X_test)
    
    # Transform features
    logger.info("Transforming features...")
    X_train_transformed = feature_engineer.fit_transform(X_train)
    X_val_transformed = feature_engineer.transform(X_val)
    X_test_transformed = feature_engineer.transform(X_test)
    
    logger.info("Feature engineering pipeline completed successfully!")
    
    return X_train_transformed, X_val_transformed, X_test_transformed, feature_engineer

def run_model_training(X_train, X_val, y_train, y_val, model_type='xgboost', use_smote=False, tune_hyperparams=False):
    """
    Run the model training pipeline.
    """
    logger.info(f"Starting model training pipeline with {model_type}...")
    
    # Initialize model trainer
    model_trainer = ModelTrainer(model_type=model_type)
    
    # Apply SMOTE if requested
    if use_smote:
        logger.info("Applying SMOTE to handle class imbalance...")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Original class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info(f"Resampled class distribution: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Perform hyperparameter tuning if requested
    if tune_hyperparams:
        logger.info("Performing hyperparameter tuning...")
        
        if model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        elif model_type == 'logistic_regression':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            param_grid = {}
        
        model_trainer.hyperparameter_tuning(
            X_train_resampled, y_train_resampled,
            X_val, y_val,
            param_grid=param_grid,
            cv=5,
            n_iter=20,
            scoring='roc_auc'
        )
    else:
        # Train the model
        logger.info("Training the model...")
        model_trainer.train(X_train_resampled, y_train_resampled)
    
    # Evaluate on validation set
    logger.info("Evaluating the model on validation set...")
    val_metrics = model_trainer.evaluate(X_val, y_val)
    
    # Save the model
    logger.info("Saving the model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_{timestamp}.joblib"
    model_path = model_trainer.save_model(model_filename)
    
    # Also save as the default model for the API
    default_model_path = os.path.join(MODELS_DIR, "churn_model.joblib")
    joblib.dump(model_trainer.model, default_model_path)
    logger.info(f"Model also saved as default model at {default_model_path}")
    
    logger.info("Model training pipeline completed successfully!")
    
    return model_trainer, val_metrics

def run_model_evaluation(model_trainer, X_test, y_test, feature_engineer=None):
    """
    Run the model evaluation pipeline.
    """
    logger.info("Starting model evaluation pipeline...")
    
    # Evaluate on test set
    logger.info("Evaluating the model on test set...")
    test_metrics = model_trainer.evaluate(X_test, y_test)
    
    # Plot evaluation metrics
    logger.info("Plotting evaluation metrics...")
    model_trainer.plot_confusion_matrix(X_test, y_test)
    model_trainer.plot_roc_curve(X_test, y_test)
    model_trainer.plot_precision_recall_curve(X_test, y_test)
    
    # Plot feature importance if feature names are available
    if feature_engineer is not None and hasattr(feature_engineer, 'feature_names'):
        logger.info("Plotting feature importance...")
        model_trainer.plot_feature_importance(feature_engineer.feature_names)
    
    logger.info("Model evaluation pipeline completed successfully!")
    
    return test_metrics

def main():
    """
    Main function to run the pipeline.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Run the entire pipeline if --all is specified
    if args.all:
        args.download = True
        args.preprocess = True
        args.train = True
        args.evaluate = True
    
    # Run data processing
    if args.download or args.preprocess:
        X_train, X_val, X_test, y_train, y_val, y_test = run_data_processing()
    else:
        # Load processed data
        from src.config import PROCESSED_DATA_DIR
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_val.csv"))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv")).iloc[:, 0]
        y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_val.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv")).iloc[:, 0]
    
    # Run feature engineering
    X_train_transformed, X_val_transformed, X_test_transformed, feature_engineer = run_feature_engineering(X_train, X_val, X_test)
    
    # Run model training
    if args.train:
        model_trainer, val_metrics = run_model_training(
            X_train_transformed, X_val_transformed, y_train, y_val,
            model_type=args.model_type,
            use_smote=args.use_smote,
            tune_hyperparams=args.tune_hyperparams
        )
    else:
        # Load the model
        from src.models.model_trainer import ModelTrainer
        model_trainer = ModelTrainer.load_model(os.path.join(MODELS_DIR, "churn_model.joblib"))
    
    # Run model evaluation
    if args.evaluate:
        test_metrics = run_model_evaluation(model_trainer, X_test_transformed, y_test, feature_engineer)
    
    logger.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    # Add missing imports
    import pandas as pd
    import numpy as np
    
    main()