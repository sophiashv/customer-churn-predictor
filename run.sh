#!/bin/bash

# Customer Churn Prediction Pipeline Runner
# This script runs the entire pipeline from data processing to API deployment

# Set up colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}      Customer Churn Prediction Pipeline Runner       ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}>> $1${NC}"
}

# Function to check if previous command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        exit 1
    fi
}

# Check if Python is installed
print_section "Checking Python installation"
python --version
check_status

# Check if required directories exist, create if not
print_section "Checking directory structure"
mkdir -p data/raw data/processed models notebooks docs tests
check_status

# Install dependencies
print_section "Installing dependencies"
pip install -r requirements.txt
check_status

# Run data processing
print_section "Processing data"
python src/run_pipeline.py --download --preprocess
check_status

# Run model training
print_section "Training model"
python src/run_pipeline.py --train --model-type xgboost --use-smote
check_status

# Run model evaluation
print_section "Evaluating model"
python src/run_pipeline.py --evaluate
check_status

# Start the API
print_section "Starting the API"
echo -e "The API will be available at http://localhost:5000"
echo -e "Press Ctrl+C to stop the API"
python src/api/app.py