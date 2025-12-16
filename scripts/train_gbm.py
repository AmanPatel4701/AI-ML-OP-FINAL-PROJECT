"""
GBM (Gradient Boosting Machine) Model Training Script

This script:
1. Loads training and validation datasets
2. Trains a GBM regression model using scikit-learn
3. Evaluates on validation and test sets
4. Logs parameters, metrics, and artifacts to MLflow
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
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
MODEL_NAME = "gbm"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GBM model for electricity forecasting")
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
    Train GBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        model: Trained GBM model
    """
    print("\nTraining GBM model...")
    
    # GBM hyperparameters
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'tol': 1e-4
    }
    
    # Train model
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    return model, params


def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True target values
        dataset_name: Name of dataset (for logging)
        
    Returns:
        metrics: Dictionary with RMSE, MAE, R²
        predictions: Array of predictions
    """
    predictions = model.predict(X)
    
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


def create_feature_importance_plot(model, feature_names, save_path):
    """
    Create feature importance plot.
    
    Args:
        model: Trained GBM model
        feature_names: List of feature names
        save_path: Path to save plot
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)  # Top 20 features
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'{MODEL_NAME.upper()} - Top 20 Feature Importance'
    )
    
    fig.update_layout(
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
        model, params = train_model(X_train, y_train, X_val, y_val)
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", MODEL_NAME)
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        
        # Evaluate on validation set
        val_metrics, y_val_pred = evaluate_model(model, X_val, y_val, "val")
        mlflow.log_metrics(val_metrics)
        
        # Evaluate on test set
        test_metrics, y_test_pred = evaluate_model(model, X_test, y_test, "test")
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
            
            # Feature importance
            importance_path = Path(tmpdir) / "feature_importance.html"
            create_feature_importance_plot(model, X_train.columns, importance_path)
            mlflow.log_artifact(str(importance_path), "plots")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, f"{MODEL_NAME}_electricity_forecast")
        
        print("\n" + "=" * 60)
        print("Model training and logging completed!")
        print("=" * 60)
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Model registered as: {MODEL_NAME}_electricity_forecast")


if __name__ == "__main__":
    main()

