"""
Example script demonstrating how to use the Customer Churn Prediction API.
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# API endpoint
API_URL = "http://localhost:5000"

def check_health():
    """
    Check if the API is running and healthy.
    """
    response = requests.get(f"{API_URL}/health")
    return response.json()

def get_model_info():
    """
    Get information about the deployed model.
    """
    response = requests.get(f"{API_URL}/model_info")
    return response.json()

def predict_single_customer(customer_data):
    """
    Make a prediction for a single customer.
    
    Args:
        customer_data: Dictionary containing customer information
        
    Returns:
        dict: Prediction result
    """
    response = requests.post(
        f"{API_URL}/predict",
        json=customer_data,
        headers={"Content-Type": "application/json"}
    )
    return response.json()

def predict_multiple_customers(customers_data):
    """
    Make predictions for multiple customers.
    
    Args:
        customers_data: List of dictionaries containing customer information
        
    Returns:
        dict: Prediction results
    """
    response = requests.post(
        f"{API_URL}/batch_predict",
        json=customers_data,
        headers={"Content-Type": "application/json"}
    )
    return response.json()

def visualize_prediction(prediction_result):
    """
    Visualize the prediction result.
    
    Args:
        prediction_result: Dictionary containing prediction result
    """
    if prediction_result['status'] != 'success':
        print(f"Error: {prediction_result.get('message', 'Unknown error')}")
        return
    
    prediction = prediction_result['prediction']
    churn_prob = prediction['churn_probability']
    churn_label = prediction['churn_label']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot churn probability as a gauge
    ax1.set_aspect('equal')
    ax1.add_patch(plt.Circle((0, 0), 1, fill=False, linewidth=2))
    ax1.add_patch(plt.Circle((0, 0), 0.8, fill=True, color='#f9f9f9'))
    
    # Add colored arc for probability
    theta1 = -90  # Start at top
    theta2 = theta1 + 180 * churn_prob  # Arc length based on probability
    
    # Choose color based on probability
    if churn_prob < 0.3:
        color = '#2ecc71'  # Green
    elif churn_prob < 0.7:
        color = '#f39c12'  # Orange
    else:
        color = '#e74c3c'  # Red
    
    ax1.add_patch(plt.matplotlib.patches.Wedge(
        (0, 0), 0.9, theta1, theta2, width=0.2, color=color
    ))
    
    # Add text
    ax1.text(0, -0.2, f"{churn_prob:.2%}", ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.text(0, -0.5, "Churn Probability", ha='center', va='center', fontsize=14)
    ax1.text(0, 0.5, churn_label, ha='center', va='center', fontsize=18, 
             fontweight='bold', color='#e74c3c' if churn_label == 'Yes' else '#2ecc71')
    
    # Remove ticks and spines
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axis('off')
    
    # Plot a simple bar chart for the second subplot
    ax2.barh(['Churn Probability'], [churn_prob], color=color)
    ax2.barh(['Retention Probability'], [1 - churn_prob], color='#3498db')
    ax2.set_xlim(0, 1)
    ax2.set_title('Probability Comparison')
    ax2.set_xlabel('Probability')
    
    # Add value labels
    ax2.text(churn_prob / 2, 0, f"{churn_prob:.2%}", ha='center', va='center', color='white')
    ax2.text(1 - churn_prob / 2, 1, f"{1-churn_prob:.2%}", ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.show()

def visualize_batch_predictions(batch_result):
    """
    Visualize batch prediction results.
    
    Args:
        batch_result: Dictionary containing batch prediction results
    """
    if batch_result['status'] != 'success':
        print(f"Error: {batch_result.get('message', 'Unknown error')}")
        return
    
    predictions = batch_result['predictions']
    
    # Create DataFrame from predictions
    df = pd.DataFrame(predictions)
    
    # Plot churn probability distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['churn_probability'], bins=20, kde=True)
    plt.title('Distribution of Churn Probabilities')
    plt.xlabel('Churn Probability')
    plt.ylabel('Count')
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.legend()
    plt.show()
    
    # Plot churn prediction counts
    plt.figure(figsize=(8, 6))
    churn_counts = df['churn_label'].value_counts()
    plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'] if 'No' in churn_counts.index else ['#e74c3c', '#2ecc71'])
    plt.title('Churn Prediction Distribution')
    plt.show()

def main():
    """
    Main function to demonstrate API usage.
    """
    print("Checking API health...")
    health_status = check_health()
    print(f"API Health: {health_status}")
    
    print("\nGetting model information...")
    model_info = get_model_info()
    print(f"Model Info: {json.dumps(model_info, indent=2)}")
    
    print("\nMaking a single customer prediction...")
    # Example customer data
    customer = {
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
    
    single_result = predict_single_customer(customer)
    print(f"Prediction Result: {json.dumps(single_result, indent=2)}")
    
    # Visualize the prediction
    visualize_prediction(single_result)
    
    print("\nMaking batch predictions...")
    # Example batch data - 5 customers with different profiles
    customers = [
        # Long-term customer with many services
        {
            "customerID": "1234-ABCD",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 72,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 118.75,
            "TotalCharges": 8550.00
        },
        # New customer with basic services
        {
            "customerID": "2345-BCDE",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
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
            "MonthlyCharges": 45.50,
            "TotalCharges": 136.50
        },
        # Mid-term customer with some services
        {
            "customerID": "3456-CDEF",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 95.30,
            "TotalCharges": 2287.20
        },
        # Senior citizen with minimal services
        {
            "customerID": "4567-DEFG",
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 48,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 55.20,
            "TotalCharges": 2649.60
        },
        # New customer with premium services
        {
            "customerID": "5678-EFGH",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 6,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 105.65,
            "TotalCharges": 633.90
        }
    ]
    
    batch_result = predict_multiple_customers(customers)
    print(f"Batch Prediction Results: {json.dumps(batch_result, indent=2)}")
    
    # Visualize batch predictions
    visualize_batch_predictions(batch_result)

if __name__ == "__main__":
    main()