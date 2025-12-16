"""
Example Client for FastAPI Electricity Forecasting API

This script demonstrates how to use the FastAPI endpoints
for making predictions with the trained models.
"""

import requests
import json
from typing import Dict, List

# API base URL
BASE_URL = "http://localhost:8000"


def predict_single(model_name: str, features: Dict) -> Dict:
    """
    Make a single prediction using a specific model.
    
    Args:
        model_name: Model to use ('xgboost', 'gbm', or 'glm')
        features: Dictionary of feature values
        
    Returns:
        Prediction response
    """
    url = f"{BASE_URL}/predict_{model_name}"
    
    payload = {
        "features": features
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()


def predict_batch(model_name: str, features_list: List[Dict]) -> Dict:
    """
    Make batch predictions using a specific model.
    
    Args:
        model_name: Model to use ('xgboost', 'gbm', or 'glm')
        features_list: List of feature dictionaries
        
    Returns:
        Batch prediction response
    """
    url = f"{BASE_URL}/predict_{model_name}"
    
    payload = {
        "features": features_list
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()


def main():
    """Example usage of the API."""
    print("=" * 60)
    print("Household Electricity Forecasting API - Example Client")
    print("=" * 60)
    
    # Example feature set (from the dataset)
    example_features = {
        "Global_reactive_power": 0.418,
        "Voltage": 234.84,
        "Global_intensity": 18.4,
        "Sub_metering_1": 0.0,
        "Sub_metering_2": 1.0,
        "Sub_metering_3": 17.0,
        "hour": 17,
        "dayofweek": 5,
        "month": 12,
        "is_weekend": 1,
        "lag_1h": 4.2,
        "rolling_mean_3h": 4.3
    }
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test single prediction with XGBoost
    print("\n2. Testing XGBoost prediction (single)...")
    try:
        result = predict_single("xgboost", example_features)
        print(json.dumps(result, indent=2))
        print(f"\nPredicted Global Active Power: {result['prediction']:.4f} kW")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test single prediction with GBM
    print("\n3. Testing GBM prediction (single)...")
    try:
        result = predict_single("gbm", example_features)
        print(json.dumps(result, indent=2))
        print(f"\nPredicted Global Active Power: {result['prediction']:.4f} kW")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test single prediction with GLM
    print("\n4. Testing GLM prediction (single)...")
    try:
        result = predict_single("glm", example_features)
        print(json.dumps(result, indent=2))
        print(f"\nPredicted Global Active Power: {result['prediction']:.4f} kW")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test batch prediction
    print("\n5. Testing batch prediction (XGBoost)...")
    batch_features = [
        example_features,
        {
            "Global_reactive_power": 0.436,
            "Voltage": 233.63,
            "Global_intensity": 23.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 16.0,
            "hour": 17,
            "dayofweek": 5,
            "month": 12,
            "is_weekend": 1,
            "lag_1h": 5.36,
            "rolling_mean_3h": 4.6
        }
    ]
    
    try:
        result = predict_batch("xgboost", batch_features)
        print(json.dumps(result, indent=2))
        print(f"\nBatch predictions: {result['predictions']}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example client completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

