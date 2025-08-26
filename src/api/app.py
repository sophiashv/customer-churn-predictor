"""
Flask API for deploying the customer churn prediction model.
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import sys
import os
import json

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, MODEL_FILE, API_HOST, API_PORT
from src.features.feature_engineering import FeatureEngineer

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the model and feature engineer
model = None
feature_engineer = None

def load_model():
    """
    Load the trained model and feature engineer.
    """
    global model, feature_engineer
    
    # Load the model
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    print("Feature engineer initialized")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    return jsonify({'status': 'ok', 'message': 'Model is loaded and ready for predictions'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    """
    # Check if model is loaded
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    # Get request data
    data = request.get_json()
    
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    try:
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Apply feature engineering
        input_data = feature_engineer.create_interaction_features(input_data)
        input_data = feature_engineer.create_service_count_feature(input_data)
        
        # Transform features
        input_transformed = feature_engineer.transform(input_data)
        
        # Make prediction
        churn_probability = model.predict_proba(input_transformed)[0, 1]
        churn_prediction = int(churn_probability >= 0.5)
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'churn_probability': float(churn_probability),
                'churn_prediction': churn_prediction,
                'churn_label': 'Yes' if churn_prediction == 1 else 'No'
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint.
    """
    # Check if model is loaded
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    # Get request data
    data = request.get_json()
    
    if not data or not isinstance(data, list):
        return jsonify({'status': 'error', 'message': 'Invalid data format. Expected a list of customer records'}), 400
    
    try:
        # Convert data to DataFrame
        input_data = pd.DataFrame(data)
        
        # Apply feature engineering
        input_data = feature_engineer.create_interaction_features(input_data)
        input_data = feature_engineer.create_service_count_feature(input_data)
        
        # Transform features
        input_transformed = feature_engineer.transform(input_data)
        
        # Make predictions
        churn_probabilities = model.predict_proba(input_transformed)[:, 1]
        churn_predictions = (churn_probabilities >= 0.5).astype(int)
        
        # Prepare response
        predictions = []
        for i, (prob, pred) in enumerate(zip(churn_probabilities, churn_predictions)):
            customer_id = data[i].get('customerID', f'customer_{i}')
            predictions.append({
                'customerID': customer_id,
                'churn_probability': float(prob),
                'churn_prediction': int(pred),
                'churn_label': 'Yes' if pred == 1 else 'No'
            })
        
        response = {
            'status': 'success',
            'predictions': predictions
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get model information.
    """
    # Check if model is loaded
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    try:
        # Get model type
        model_type = type(model).__name__
        
        # Get model parameters
        params = model.get_params()
        
        # Convert numpy values to Python native types for JSON serialization
        for key, value in params.items():
            if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                params[key] = int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                params[key] = float(value)
            elif isinstance(value, (np.ndarray,)):
                params[key] = value.tolist()
        
        # Prepare response
        response = {
            'status': 'success',
            'model_info': {
                'model_type': model_type,
                'model_parameters': params,
                'model_file': MODEL_FILE
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """
    Web interface for the customer churn prediction model.
    """
    return render_template('index.html')

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """
    API documentation endpoint.
    """
    docs = {
        'api_name': 'Customer Churn Prediction API',
        'description': 'API for predicting customer churn',
        'endpoints': [
            {
                'path': '/',
                'method': 'GET',
                'description': 'Web interface for the customer churn prediction model'
            },
            {
                'path': '/api/docs',
                'method': 'GET',
                'description': 'API documentation'
            },
            {
                'path': '/health',
                'method': 'GET',
                'description': 'Health check endpoint'
            },
            {
                'path': '/predict',
                'method': 'POST',
                'description': 'Make a prediction for a single customer',
                'request_format': {
                    'customerID': 'string',
                    'gender': 'string',
                    'SeniorCitizen': 'int',
                    'Partner': 'string',
                    'Dependents': 'string',
                    'tenure': 'int',
                    'PhoneService': 'string',
                    'MultipleLines': 'string',
                    'InternetService': 'string',
                    'OnlineSecurity': 'string',
                    'OnlineBackup': 'string',
                    'DeviceProtection': 'string',
                    'TechSupport': 'string',
                    'StreamingTV': 'string',
                    'StreamingMovies': 'string',
                    'Contract': 'string',
                    'PaperlessBilling': 'string',
                    'PaymentMethod': 'string',
                    'MonthlyCharges': 'float',
                    'TotalCharges': 'float'
                }
            },
            {
                'path': '/batch_predict',
                'method': 'POST',
                'description': 'Make predictions for multiple customers',
                'request_format': 'List of customer records'
            },
            {
                'path': '/model_info',
                'method': 'GET',
                'description': 'Get model information'
            }
        ]
    }
    
    return jsonify(docs), 200

def main():
    """
    Main function to run the Flask API.
    """
    # Load the model
    load_model()
    
    # Run the Flask app
    app.run(host=API_HOST, port=API_PORT, debug=False)

if __name__ == "__main__":
    main()