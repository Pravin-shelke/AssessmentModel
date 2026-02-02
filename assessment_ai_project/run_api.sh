#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# Add current directory to PYTHONPATH so imports work
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Assessment AI API Server on port 5001..."
echo "If this fails, please ensure you have installed dependencies:"
echo "pip install flask flask-cors pandas xgboost scikit-learn numpy"
echo "---------------------------------------------------"

python3 src/api_server.py
