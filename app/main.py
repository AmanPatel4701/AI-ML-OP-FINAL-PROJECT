"""
FastAPI Application for Household Electricity Forecasting

This application:
1. Loads three registered models from MLflow
2. Provides prediction endpoints for each model
3. Validates input data
4. Returns predictions with metadata
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Union
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Household Electricity Forecasting API",
    description="API for predicting household electricity consumption using ML models",
    version="1.0.0"
)

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME_XGBOOST = "xgboost_electricity_forecast"
MODEL_REGISTRY_NAME_GBM = "gbm_electricity_forecast"
MODEL_REGISTRY_NAME_GLM = "glm_electricity_forecast"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables to store loaded models
models = {}
scalers = {}


class ElectricityFeatures(BaseModel):
    """Pydantic model for input features validation."""
    Global_reactive_power: float = Field(..., description="Global reactive power (kW)")
    Voltage: float = Field(..., ge=100, le=300, description="Voltage (V)")
    Global_intensity: float = Field(..., ge=0, description="Global intensity (A)")
    Sub_metering_1: float = Field(..., ge=0, description="Sub-metering 1 (Wh)")
    Sub_metering_2: float = Field(..., ge=0, description="Sub-metering 2 (Wh)")
    Sub_metering_3: float = Field(..., ge=0, description="Sub-metering 3 (Wh)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0 or 1)")
    lag_1h: float = Field(..., ge=0, description="Lag feature: power 1 hour ago (kW)")
    rolling_mean_3h: float = Field(..., ge=0, description="Rolling mean of last 3 hours (kW)")

    @field_validator('*')
    @classmethod
    def validate_not_nan(cls, v):
        """Ensure no NaN values."""
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            raise ValueError("NaN or Inf values are not allowed")
        return v


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: ElectricityFeatures


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[ElectricityFeatures]


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float
    model_name: str
    model_version: str
    timestamp: str
    input_features: dict


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[float]
    model_name: str
    model_version: str
    timestamp: str
    count: int


def load_model_from_registry(model_name: str, model_key: str):
    """
    Load a model from MLflow Model Registry.
    
    Args:
        model_name: Name of the model in registry
        model_key: Key to store model in global dict
        
    Returns:
        model: Loaded MLflow model
    """
    try:
        # Get the latest version of the model
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        
        if not latest_version:
            raise ValueError(f"No versions found for model {model_name}")
        
        # Get the latest version
        version = latest_version[0].version
        model_uri = f"models:/{model_name}/{version}"
        
        print(f"Loading {model_name} version {version} from {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Try to load scaler if it exists (for GLM)
        # Note: Scaler is logged separately, we'll try to load it from the run
        try:
            run_id = latest_version[0].run_id
            # Try to load scaler from artifacts
            scaler_uri = f"runs:/{run_id}/scaler"
            scaler = mlflow.sklearn.load_model(scaler_uri)
            scalers[model_key] = scaler
            print(f"Loaded scaler for {model_key}")
        except Exception as e:
            print(f"Note: Scaler not found for {model_key} (this is OK for non-GLM models): {e}")
            scalers[model_key] = None
        
        return model, version
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model {model_name}: {str(e)}"
        )


def initialize_models():
    """Initialize all models on startup."""
    print("Initializing models from MLflow Model Registry...")
    
    try:
        # Load XGBoost model
        model, version = load_model_from_registry(MODEL_REGISTRY_NAME_XGBOOST, "xgboost")
        models["xgboost"] = model
        models["xgboost_version"] = version
        print(f"Loaded XGBoost model (version {version})")
    except Exception as e:
        print(f"✗ Failed to load XGBoost model: {e}")
        models["xgboost"] = None
    
    try:
        # Load GBM model
        model, version = load_model_from_registry(MODEL_REGISTRY_NAME_GBM, "gbm")
        models["gbm"] = model
        models["gbm_version"] = version
        print(f"Loaded GBM model (version {version})")
    except Exception as e:
        print(f"✗ Failed to load GBM model: {e}")
        models["gbm"] = None
    
    try:
        # Load GLM model
        model, version = load_model_from_registry(MODEL_REGISTRY_NAME_GLM, "glm")
        models["glm"] = model
        models["glm_version"] = version
        print(f"Loaded GLM model (version {version})")
    except Exception as e:
        print(f"✗ Failed to load GLM model: {e}")
        models["glm"] = None
    
    print("Model initialization completed!")


@app.on_event("startup")
async def startup_event():
    """Initialize models when the application starts."""
    initialize_models()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Household Electricity Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "predict_xgboost": "/predict_xgboost",
            "predict_gbm": "/predict_gbm",
            "predict_glm": "/predict_glm",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = {
        "xgboost": models.get("xgboost") is not None,
        "gbm": models.get("gbm") is not None,
        "glm": models.get("glm") is not None
    }
    
    return {
        "status": "healthy",
        "models_loaded": model_status,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }


def prepare_features_df(features: Union[ElectricityFeatures, List[ElectricityFeatures]]) -> pd.DataFrame:
    """
    Convert features to DataFrame in the correct order.
    
    Args:
        features: Single or list of feature objects
        
    Returns:
        DataFrame with features in correct order
    """
    if isinstance(features, list):
        data = [f.model_dump() for f in features]
    else:
        data = [features.model_dump()]
    
    df = pd.DataFrame(data)
    
    # Ensure correct column order (matching training data)
    expected_columns = [
        'Global_reactive_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'hour', 'dayofweek', 'month', 'is_weekend',
        'lag_1h', 'rolling_mean_3h'
    ]
    
    # Reorder columns
    df = df[expected_columns]
    
    return df


def predict_with_model(model_key: str, features_df: pd.DataFrame, scaler=None):
    """
    Make predictions using a specific model.
    
    Args:
        model_key: Key of the model to use
        features_df: DataFrame with features
        scaler: Optional scaler for GLM model
        
    Returns:
        Predictions array
    """
    model = models.get(model_key)
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {model_key} is not loaded. Please check model initialization."
        )
    
    # Scale features if scaler is provided (for GLM)
    if scaler is not None:
        features_df = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns
        )
    
    # Make predictions
    predictions = model.predict(features_df)
    
    return predictions


@app.post("/predict_xgboost", response_model=Union[PredictionResponse, BatchPredictionResponse])
async def predict_xgboost(request: Union[PredictionRequest, BatchPredictionRequest]):
    """
    Predict using XGBoost model.
    
    Accepts single or batch predictions.
    """
    if models.get("xgboost") is None:
        raise HTTPException(status_code=503, detail="XGBoost model is not loaded")
    
    # Prepare features
    if isinstance(request, PredictionRequest):
        features_df = prepare_features_df(request.features)
        is_batch = False
    else:
        features_df = prepare_features_df(request.features)
        is_batch = True
    
    # Make predictions
    predictions = predict_with_model("xgboost", features_df)
    
    # Prepare response
    timestamp = datetime.now().isoformat()
    model_version = models.get("xgboost_version", "unknown")
    
    if is_batch:
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            model_name="xgboost",
            model_version=model_version,
            timestamp=timestamp,
            count=len(predictions)
        )
    else:
        return PredictionResponse(
            prediction=float(predictions[0]),
            model_name="xgboost",
            model_version=model_version,
            timestamp=timestamp,
            input_features=request.features.model_dump()
        )


@app.post("/predict_gbm", response_model=Union[PredictionResponse, BatchPredictionResponse])
async def predict_gbm(request: Union[PredictionRequest, BatchPredictionRequest]):
    """
    Predict using GBM model.
    
    Accepts single or batch predictions.
    """
    if models.get("gbm") is None:
        raise HTTPException(status_code=503, detail="GBM model is not loaded")
    
    # Prepare features
    if isinstance(request, PredictionRequest):
        features_df = prepare_features_df(request.features)
        is_batch = False
    else:
        features_df = prepare_features_df(request.features)
        is_batch = True
    
    # Make predictions
    predictions = predict_with_model("gbm", features_df)
    
    # Prepare response
    timestamp = datetime.now().isoformat()
    model_version = models.get("gbm_version", "unknown")
    
    if is_batch:
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            model_name="gbm",
            model_version=model_version,
            timestamp=timestamp,
            count=len(predictions)
        )
    else:
        return PredictionResponse(
            prediction=float(predictions[0]),
            model_name="gbm",
            model_version=model_version,
            timestamp=timestamp,
            input_features=request.features.model_dump()
        )


@app.post("/predict_glm", response_model=Union[PredictionResponse, BatchPredictionResponse])
async def predict_glm(request: Union[PredictionRequest, BatchPredictionRequest]):
    """
    Predict using GLM model.
    
    Accepts single or batch predictions.
    Note: GLM requires feature scaling.
    """
    if models.get("glm") is None:
        raise HTTPException(status_code=503, detail="GLM model is not loaded")
    
    # Get scaler for GLM
    scaler = scalers.get("glm")
    if scaler is None:
        raise HTTPException(status_code=503, detail="GLM scaler is not loaded")
    
    # Prepare features
    if isinstance(request, PredictionRequest):
        features_df = prepare_features_df(request.features)
        is_batch = False
    else:
        features_df = prepare_features_df(request.features)
        is_batch = True
    
    # Make predictions (with scaling)
    predictions = predict_with_model("glm", features_df, scaler=scaler)
    
    # Prepare response
    timestamp = datetime.now().isoformat()
    model_version = models.get("glm_version", "unknown")
    
    if is_batch:
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            model_name="glm",
            model_version=model_version,
            timestamp=timestamp,
            count=len(predictions)
        )
    else:
        return PredictionResponse(
            prediction=float(predictions[0]),
            model_name="glm",
            model_version=model_version,
            timestamp=timestamp,
            input_features=request.features.model_dump()
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

