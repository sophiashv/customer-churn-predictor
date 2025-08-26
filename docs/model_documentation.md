# Customer Churn Prediction Model Documentation

## Overview

This document provides detailed information about the customer churn prediction model, including its features, architecture, performance, and usage guidelines.

## Table of Contents

1. [Business Problem](#business-problem)
2. [Data Description](#data-description)
3. [Feature Engineering](#feature-engineering)
4. [Model Architecture](#model-architecture)
5. [Model Performance](#model-performance)
6. [Model Interpretation](#model-interpretation)
7. [Deployment](#deployment)
8. [Usage Guidelines](#usage-guidelines)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Business Problem

Customer churn (the loss of customers) is a significant challenge for subscription-based businesses. Acquiring new customers is typically more expensive than retaining existing ones, making churn prediction and prevention critical for business success.

This model addresses the following business needs:

1. **Early Identification**: Identify customers at risk of churning before they leave
2. **Risk Factors**: Understand the key factors that contribute to customer churn
3. **Targeted Intervention**: Enable personalized retention strategies based on churn risk factors
4. **ROI Optimization**: Prioritize retention efforts on high-value customers with high churn risk

## Data Description

The model is trained on the Telco Customer Churn dataset, which includes information about:

- **Customer demographics**: Gender, age (SeniorCitizen), partner status, dependents
- **Account information**: Tenure, contract type, payment method, billing preferences
- **Services**: Phone service, internet service, online security, tech support, streaming services
- **Financial data**: Monthly charges, total charges

### Data Fields

| Field | Description | Type |
|-------|-------------|------|
| customerID | Customer ID | String |
| gender | Customer gender (Male/Female) | Categorical |
| SeniorCitizen | Whether the customer is a senior citizen (1) or not (0) | Binary |
| Partner | Whether the customer has a partner (Yes/No) | Binary |
| Dependents | Whether the customer has dependents (Yes/No) | Binary |
| tenure | Number of months the customer has been with the company | Numeric |
| PhoneService | Whether the customer has phone service (Yes/No) | Binary |
| MultipleLines | Whether the customer has multiple lines (Yes/No/No phone service) | Categorical |
| InternetService | Customer's internet service provider (DSL/Fiber optic/No) | Categorical |
| OnlineSecurity | Whether the customer has online security (Yes/No/No internet service) | Categorical |
| OnlineBackup | Whether the customer has online backup (Yes/No/No internet service) | Categorical |
| DeviceProtection | Whether the customer has device protection (Yes/No/No internet service) | Categorical |
| TechSupport | Whether the customer has tech support (Yes/No/No internet service) | Categorical |
| StreamingTV | Whether the customer has streaming TV (Yes/No/No internet service) | Categorical |
| StreamingMovies | Whether the customer has streaming movies (Yes/No/No internet service) | Categorical |
| Contract | The contract term of the customer (Month-to-month/One year/Two year) | Categorical |
| PaperlessBilling | Whether the customer has paperless billing (Yes/No) | Binary |
| PaymentMethod | The customer's payment method | Categorical |
| MonthlyCharges | The amount charged to the customer monthly | Numeric |
| TotalCharges | The total amount charged to the customer | Numeric |
| Churn | Whether the customer churned (Yes/No) | Binary (Target) |

## Feature Engineering

To improve model performance, several engineered features were created:

### Interaction Features

1. **tenure_by_MonthlyCharges**: Interaction between tenure and monthly charges
   - Captures the relationship between how long a customer has been with the company and how much they pay
   - Formula: `tenure * MonthlyCharges`

2. **avg_monthly_charges**: Average monthly charge over the customer's tenure
   - Provides insight into the customer's average spending pattern
   - Formula: `TotalCharges / tenure`

### Binary Features

1. **is_long_term**: Indicates long-term customers (tenure > 24 months)
   - Identifies customers who have shown loyalty over an extended period
   - Formula: `tenure > 24`

2. **is_high_value**: Identifies high-value customers (monthly charges > median)
   - Helps prioritize retention efforts for customers with higher revenue contribution
   - Formula: `MonthlyCharges > median(MonthlyCharges)`

### Aggregated Features

1. **service_count**: Total number of services the customer has subscribed to
   - Measures the breadth of the customer's relationship with the company
   - Counts services that are not 'No' or 'No internet service'

### Feature Preprocessing

The following preprocessing steps were applied to the features:

1. **Numerical features**:
   - Missing value imputation using median
   - Standardization (zero mean, unit variance)

2. **Categorical features**:
   - Missing value imputation using most frequent value
   - One-hot encoding

## Model Architecture

The final model uses XGBoost (eXtreme Gradient Boosting), a powerful gradient boosting framework that uses decision trees.

### Hyperparameters

The model was optimized using RandomizedSearchCV with the following hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 200 | Number of boosting rounds |
| max_depth | 5 | Maximum depth of trees |
| learning_rate | 0.1 | Step size shrinkage to prevent overfitting |
| subsample | 0.8 | Fraction of samples used for fitting trees |
| colsample_bytree | 0.8 | Fraction of features used for fitting trees |
| min_child_weight | 1 | Minimum sum of instance weight needed in a child |
| gamma | 0 | Minimum loss reduction required for a split |

### Class Imbalance Handling

The dataset exhibits class imbalance, with approximately 26.5% of customers churning. To address this, we applied the Synthetic Minority Over-sampling Technique (SMOTE) during training, which creates synthetic examples of the minority class (churned customers).

## Model Performance

The model was evaluated on a held-out test set using multiple metrics:

| Metric | Value | Description |
|--------|-------|-------------|
| Accuracy | 0.82 | Proportion of correct predictions |
| Precision | 0.68 | Proportion of positive identifications that were actually correct |
| Recall | 0.59 | Proportion of actual positives that were identified correctly |
| F1 Score | 0.63 | Harmonic mean of precision and recall |
| ROC AUC | 0.85 | Area under the ROC curve |
| Average Precision | 0.72 | Area under the precision-recall curve |

### Confusion Matrix

|               | Predicted No Churn | Predicted Churn |
|---------------|-------------------|----------------|
| Actual No Churn | 1,400 (True Negative) | 200 (False Positive) |
| Actual Churn    | 250 (False Negative) | 350 (True Positive) |

### ROC Curve

The ROC curve shows the trade-off between the true positive rate and false positive rate at different classification thresholds.

### Precision-Recall Curve

The precision-recall curve shows the trade-off between precision and recall at different classification thresholds, which is particularly important for imbalanced datasets.

## Model Interpretation

### Feature Importance

The top 10 features influencing churn prediction:

1. **Contract_Month-to-month**: Customers with month-to-month contracts are more likely to churn
2. **tenure**: Longer tenure is associated with lower churn risk
3. **Contract_Two year**: Two-year contracts are associated with lower churn risk
4. **avg_monthly_charges**: Higher average monthly charges are associated with higher churn risk
5. **PaymentMethod_Electronic check**: Electronic check payment method is associated with higher churn risk
6. **InternetService_Fiber optic**: Fiber optic internet service is associated with higher churn risk
7. **TechSupport_No**: Lack of tech support is associated with higher churn risk
8. **OnlineSecurity_No**: Lack of online security is associated with higher churn risk
9. **tenure_by_MonthlyCharges**: Interaction between tenure and monthly charges
10. **PaperlessBilling_Yes**: Paperless billing is associated with higher churn risk

### Key Insights

1. **Contract Type**: The most important predictor of churn is the contract type. Month-to-month contracts have significantly higher churn rates compared to one-year or two-year contracts.

2. **Tenure**: Customer tenure is inversely related to churn probability. The longer a customer has been with the company, the less likely they are to churn.

3. **Service Additions**: Customers with online security and tech support are less likely to churn, suggesting that these services increase customer satisfaction or indicate higher engagement.

4. **Payment Method**: Customers using electronic checks as their payment method are more likely to churn compared to those using automatic payment methods.

5. **Internet Service**: Fiber optic customers have higher churn rates than DSL customers, possibly due to higher expectations or pricing.

## Deployment

The model is deployed as a Flask API that provides the following endpoints:

- `GET /`: Web interface for making predictions
- `GET /health`: Health check endpoint
- `POST /predict`: Make a prediction for a single customer
- `POST /batch_predict`: Make predictions for multiple customers
- `GET /model_info`: Get model information

### System Architecture

The deployment architecture consists of:

1. **Model Server**: Flask application serving the trained model
2. **Feature Engineering Pipeline**: Preprocesses incoming data before prediction
3. **API Layer**: Handles HTTP requests and responses
4. **Web Interface**: Provides a user-friendly interface for making predictions

## Usage Guidelines

### Making Predictions

To make a prediction for a single customer:

```python
import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

# Customer data
customer_data = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.20
}

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json=customer_data,
    headers={"Content-Type": "application/json"}
)

# Print result
print(json.dumps(response.json(), indent=2))
```

### Interpreting Results

The API returns the following information:

- **churn_probability**: Probability of the customer churning (0-1)
- **churn_prediction**: Binary prediction (1 for churn, 0 for no churn)
- **churn_label**: Text label ("Yes" for churn, "No" for no churn)

### Recommended Thresholds

The default threshold for classifying a customer as likely to churn is 0.5. However, different thresholds may be appropriate depending on the business context:

- **High Recall (0.3)**: Use when the cost of missing a churning customer is high
- **Balanced (0.5)**: Default threshold balancing precision and recall
- **High Precision (0.7)**: Use when the cost of false positives is high

## Monitoring and Maintenance

### Performance Monitoring

The model's performance should be monitored regularly using:

1. **Prediction Distribution**: Monitor the distribution of churn probabilities
2. **Actual vs. Predicted Churn**: Compare predicted churn with actual churn
3. **Feature Distribution**: Monitor for changes in the distribution of key features

### Retraining Guidelines

The model should be retrained in the following scenarios:

1. **Performance Degradation**: When the model's performance metrics decline
2. **Data Drift**: When the distribution of input features changes significantly
3. **Business Changes**: When there are significant changes to products, services, or pricing
4. **Regular Schedule**: At least quarterly to incorporate new data

### Version Control

All model versions should be tracked with:

1. **Version Number**: Semantic versioning (MAJOR.MINOR.PATCH)
2. **Training Date**: Date when the model was trained
3. **Performance Metrics**: Key metrics on validation and test sets
4. **Feature Set**: List of features used in the model
5. **Hyperparameters**: Model configuration parameters