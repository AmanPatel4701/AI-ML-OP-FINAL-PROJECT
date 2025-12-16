"""
GLM (Generalized Linear Model) Model Training Script

This script:
1. Loads training and validation datasets
2. Trains a GLM regression model using scikit-learn (ElasticNet)
3. Evaluates on validation and test sets
4. Logs parameters, metrics, and artifacts to MLflow
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import warnings

warnings.filterwarnings('ignore')

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validate.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

# MLflow experiment name
EXPERIMENT_NAME = "household-electricity-regression"
MODEL_NAME = "glm"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GLM model for electricity forecasting")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)"
    )
    return parser.parse_args()


def load_data():
    """
    Load training, validation, and test datasets.
    
    Returns:
        train_df, validate_df, test_df: DataFrames with features and target
    """
    print("Loading datasets...")
    
    train_df = pd.read_csv(TRAIN_DATA_PATH, index_col=0, parse_dates=True)
    validate_df = pd.read_csv(VALIDATE_DATA_PATH, index_col=0, parse_dates=True)
    test_df = pd.read_csv(TEST_DATA_PATH, index_col=0, parse_dates=True)
    
    print(f"Train set: {len(train_df):,} rows")
    print(f"Validation set: {len(validate_df):,} rows")
    print(f"Test set: {len(test_df):,} rows")
    
    return train_df, validate_df, test_df


def prepare_features(df, target_column="Global_active_power"):
    """
    Separate features and target.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def train_model(X_train, y_train, X_val, y_val):
    """
    Train GLM (ElasticNet) model with feature scaling.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        model: Trained GLM model
        scaler: Fitted StandardScaler
    """
    print("\nTraining GLM (ElasticNet) model...")
    
    # Scale features (important for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ElasticNet hyperparameters (GLM with L1+L2 regularization)
    params = {
        'alpha': 0.1,
        'l1_ratio': 0.5,  # Balance between L1 and L2 (0.5 = equal)
        'max_iter': 2000,
        'tol': 1e-4,
        'random_state': 42
    }
    
    # Train model
    model = ElasticNet(**params)
    model.fit(X_train_scaled, y_train)
    
    print("Model training completed!")
    
    return model, scaler, params


def evaluate_model(model, scaler, X, y, dataset_name):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        X: Features
        y: True target values
        dataset_name: Name of dataset (for logging)
        
    Returns:
        metrics: Dictionary with RMSE, MAE, R²
        predictions: Array of predictions
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    metrics = {
        f'{dataset_name}_rmse': rmse,
        f'{dataset_name}_mae': mae,
        f'{dataset_name}_r2': r2
    }
    
    print(f"\n{dataset_name.upper()} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics, predictions


def create_prediction_plot(y_true, y_pred, dataset_name, save_path):
    """
    Create prediction vs actual scatter plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dataset_name: Name of dataset
        save_path: Path to save plot
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(size=3, opacity=0.5),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=f'{MODEL_NAME.upper()} - Prediction vs Actual ({dataset_name})',
        xaxis_title='Actual Global Active Power (kW)',
        yaxis_title='Predicted Global Active Power (kW)',
        width=800,
        height=600
    )
    
    fig.write_html(str(save_path))


def create_residual_plot(y_true, y_pred, dataset_name, save_path):
    """
    Create residual plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dataset_name: Name of dataset
        save_path: Path to save plot
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(size=3, opacity=0.5),
        name='Residuals'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f'{MODEL_NAME.upper()} - Residual Plot ({dataset_name})',
        xaxis_title='Predicted Global Active Power (kW)',
        yaxis_title='Residuals (Actual - Predicted)',
        width=800,
        height=600
    )
    
    fig.write_html(str(save_path))


def create_coefficient_plot(model, feature_names, save_path):
    """
    Create coefficient plot (for linear models, coefficients show feature importance).
    
    Args:
        model: Trained GLM model
        feature_names: List of feature names
        save_path: Path to save plot
    """
    # Get coefficients (absolute value for importance)
    coefficients = np.abs(model.coef_)
    
    # Create DataFrame and sort
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False).head(20)  # Top 20 features
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=coef_df['coefficient'],
        y=coef_df['feature'],
        orientation='h',
        name='Coefficient Magnitude'
    ))
    
    fig.update_layout(
        title=f'{MODEL_NAME.upper()} - Top 20 Feature Coefficients (Importance)',
        xaxis_title='Absolute Coefficient Value',
        yaxis_title='Feature',
        width=800,
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.write_html(str(save_path))


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 60)
    print(f"Training {MODEL_NAME.upper()} Model")
    print("=" * 60)
    print(f"MLflow Tracking URI: {args.tracking_uri}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    train_df, validate_df, test_df = load_data()
    
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(validate_df)
    X_test, y_test = prepare_features(test_df)
    
    print(f"\nFeatures: {list(X_train.columns)}")
    print(f"Number of features: {len(X_train.columns)}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{MODEL_NAME}_run"):
        # Train model
        model, scaler, params = train_model(X_train, y_train, X_val, y_val)
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", MODEL_NAME)
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        
        # Evaluate on validation set
        val_metrics, y_val_pred = evaluate_model(model, scaler, X_val, y_val, "val")
        mlflow.log_metrics(val_metrics)
        
        # Evaluate on test set
        test_metrics, y_test_pred = evaluate_model(model, scaler, X_test, y_test, "test")
        mlflow.log_metrics(test_metrics)
        
        # Create and log plots
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prediction vs actual (validation)
            val_pred_path = Path(tmpdir) / "val_prediction_plot.html"
            create_prediction_plot(y_val, y_val_pred, "Validation", val_pred_path)
            mlflow.log_artifact(str(val_pred_path), "plots")
            
            # Prediction vs actual (test)
            test_pred_path = Path(tmpdir) / "test_prediction_plot.html"
            create_prediction_plot(y_test, y_test_pred, "Test", test_pred_path)
            mlflow.log_artifact(str(test_pred_path), "plots")
            
            # Residual plot (validation)
            val_residual_path = Path(tmpdir) / "val_residual_plot.html"
            create_residual_plot(y_val, y_val_pred, "Validation", val_residual_path)
            mlflow.log_artifact(str(val_residual_path), "plots")
            
            # Coefficient plot (instead of feature importance for linear models)
            coef_path = Path(tmpdir) / "coefficient_plot.html"
            create_coefficient_plot(model, X_train.columns, coef_path)
            mlflow.log_artifact(str(coef_path), "plots")
        
        # Log model and scaler separately
        # The scaler is needed for inference, so we log it as an artifact
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Also log scaler path for easier retrieval
        mlflow.log_param("scaler_path", "scaler")
        
        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, f"{MODEL_NAME}_electricity_forecast")
        
        print("\n" + "=" * 60)
        print("Model training and logging completed!")
        print("=" * 60)
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Model registered as: {MODEL_NAME}_electricity_forecast")
        print("\nNote: GLM model requires scaler. Both are logged separately.")


if __name__ == "__main__":
    main()

