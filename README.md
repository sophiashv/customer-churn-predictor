# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.2-red.svg)

A complete machine learning project that predicts customer churn using advanced ML techniques. This project demonstrates a full ML workflow from data exploration to model deployment.

![Churn Prediction Demo](https://github.com/sophiashv/project-X/raw/main/docs/images/churn_prediction_demo.png)

## Quick Demo

Try the live demo: [Customer Churn Predictor](https://example.com/churn-predictor)

*Note: This project is part of a machine learning portfolio showcasing end-to-end ML development skills.*

## Project Overview

Customer churn prediction is a critical business problem for subscription-based companies. This project builds a machine learning model to predict which customers are likely to churn, allowing businesses to take proactive retention measures.

### Key Features

- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Engineering**: Creation of meaningful features to improve model performance
- **Multiple ML Models**: Implementation of various algorithms (XGBoost, Random Forest, etc.)
- **Hyperparameter Tuning**: Optimization of model parameters for best performance
- **Class Imbalance Handling**: SMOTE implementation for balanced training
- **Model Evaluation**: Thorough evaluation using multiple metrics
- **API Development**: Flask API for real-time predictions
- **Web Interface**: User-friendly interface for making predictions

## Project Structure

```
├── data/               # Data files
│   ├── raw/            # Original, immutable data
│   └── processed/      # Cleaned, transformed data
├── notebooks/          # Jupyter notebooks for exploration and development
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   └── 02_Model_Development.ipynb
├── src/                # Source code
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature engineering code
│   ├── models/         # Model training and evaluation
│   ├── visualization/  # Visualization utilities
│   ├── api/            # API for model deployment
│   └── config.py       # Configuration settings
├── models/             # Saved model files
├── docs/               # Documentation
└── tests/              # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Processing

Process the raw data and prepare it for modeling:

```bash
python src/run_pipeline.py --download --preprocess
```

### Model Training

Train the churn prediction model:

```bash
python src/run_pipeline.py --train --model-type xgboost --use-smote --tune-hyperparams
```

Available model types:
- `xgboost` (default)
- `random_forest`
- `logistic_regression`

### Model Evaluation

Evaluate the trained model:

```bash
python src/run_pipeline.py --evaluate
```

### Run Complete Pipeline

Run the entire pipeline from data processing to evaluation:

```bash
python src/run_pipeline.py --all
```

### Start the API

Start the Flask API for making predictions:

```bash
python src/api/app.py
```

The API will be available at `http://localhost:5000`.

## API Documentation

### Endpoints

- `GET /`: Web interface for making predictions
- `GET /health`: Health check endpoint
- `POST /predict`: Make a prediction for a single customer
- `POST /batch_predict`: Make predictions for multiple customers
- `GET /model_info`: Get model information

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 72,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 56.95,
    "TotalCharges": 4100.40
  }'
```

## Model Performance

The XGBoost model achieves the following performance on the test set:

- **Accuracy**: 0.82
- **Precision**: 0.68
- **Recall**: 0.59
- **F1 Score**: 0.63
- **ROC AUC**: 0.85
- **Average Precision**: 0.72

## Key Insights

1. **Contract Type**: Month-to-month contracts have significantly higher churn rates
2. **Tenure**: Longer-tenured customers are less likely to churn
3. **Payment Method**: Electronic check payment method is associated with higher churn
4. **Services**: Customers without online security and tech support are more likely to churn
5. **Internet Service**: Fiber optic customers have higher churn rates than DSL customers

## Future Improvements

- Implement more advanced feature engineering techniques
- Explore deep learning models for churn prediction
- Add model explainability using SHAP values
- Develop a more comprehensive dashboard for business users
- Implement A/B testing for retention strategies

## License

[MIT License](LICENSE)

## Contact

For questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com).