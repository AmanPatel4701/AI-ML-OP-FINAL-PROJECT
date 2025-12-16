"""
Model Drift Analysis Script

This script:
1. Uses the last 3-6 months of data as "production" data
2. Evaluates model performance on this newer window
3. Uses Evidently AI to detect data drift and performance drift
4. Generates HTML reports
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionErrorDistribution,
    RegressionAbsPercentageErrorPlot
)
import argparse

warnings.filterwarnings('ignore')

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "drift"

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME = "xgboost_electricity_forecast"  # Use best model or all models

# Create reports directory
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze model drift")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_electricity_forecast",
        help="Model name in MLflow registry"
    )
    parser.add_argument(
        "--production-months",
        type=int,
        default=6,
        help="Number of months to use as production data (default: 6)"
    )
    return parser.parse_args()


def load_data():
    """
    Load training and test datasets.
    
    Returns:
        train_df, test_df: DataFrames
    """
    print("Loading datasets...")
    
    train_df = pd.read_csv(TRAIN_DATA_PATH, index_col=0, parse_dates=True)
    test_df = pd.read_csv(TEST_DATA_PATH, index_col=0, parse_dates=True)
    
    print(f"Training set: {len(train_df):,} rows")
    print(f"  Date range: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test set: {len(test_df):,} rows")
    print(f"  Date range: {test_df.index.min()} to {test_df.index.max()}")
    
    return train_df, test_df


def extract_production_data(test_df, months=6):
    """
    Extract the last N months as production data.
    
    Args:
        test_df: Test DataFrame
        months: Number of months to extract
        
    Returns:
        production_df: Production data DataFrame
    """
    print(f"\nExtracting last {months} months as production data...")
    
    # Sort by date
    test_df = test_df.sort_index()
    
    # Get the cutoff date
    cutoff_date = test_df.index.max() - pd.DateOffset(months=months)
    
    # Extract production data (last N months)
    production_df = test_df[test_df.index >= cutoff_date].copy()
    
    print(f"Production data: {len(production_df):,} rows")
    print(f"  Date range: {production_df.index.min()} to {production_df.index.max()}")
    
    return production_df


def load_model_from_registry(model_name):
    """
    Load model from MLflow Model Registry.
    
    Args:
        model_name: Name of the model in registry
        
    Returns:
        model: Loaded MLflow model
    """
    print(f"\nLoading model {model_name} from MLflow...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        
        if not latest_version:
            raise ValueError(f"No versions found for model {model_name}")
        
        version = latest_version[0].version
        model_uri = f"models:/{model_name}/{version}"
        
        print(f"Loading version {version} from {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


def generate_predictions(model, df):
    """
    Generate predictions for a dataset.
    
    Args:
        model: Trained model
        df: DataFrame with features
        
    Returns:
        predictions: Array of predictions
    """
    # Separate features and target
    target_col = "Global_active_power"
    X = df.drop(columns=[target_col])
    
    # Generate predictions
    predictions = model.predict(X)
    
    return predictions


def create_data_drift_report(reference_df, current_df):
    """
    Create data drift report using Evidently.
    
    Args:
        reference_df: Reference dataset (training data)
        current_df: Current dataset (production data)
        
    Returns:
        report: Evidently report
    """
    print("\nGenerating Data Drift Report...")
    
    # Define column mapping
    target_col = "Global_active_power"
    feature_columns = [col for col in reference_df.columns if col != target_col]
    
    column_mapping = ColumnMapping(
        target=target_col,
        numerical_features=feature_columns,
        prediction=None  # We'll add predictions separately
    )
    
    # Create report
    report = Report(metrics=[
        DataDriftTable(),
    ])
    
    # Generate report
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    
    return report


def create_performance_report(reference_df, current_df, reference_pred, current_pred):
    """
    Create regression performance report using Evidently.
    
    Args:
        reference_df: Reference dataset
        current_df: Current dataset
        reference_pred: Predictions on reference data
        current_pred: Predictions on current data
        
    Returns:
        report: Evidently report
    """
    print("\nGenerating Regression Performance Report...")
    
    # Add predictions to dataframes
    reference_with_pred = reference_df.copy()
    reference_with_pred['prediction'] = reference_pred
    
    current_with_pred = current_df.copy()
    current_with_pred['prediction'] = current_pred
    
    # Define column mapping
    target_col = "Global_active_power"
    feature_columns = [col for col in reference_df.columns if col != target_col]
    
    column_mapping = ColumnMapping(
        target=target_col,
        prediction="prediction",
        numerical_features=feature_columns
    )
    
    # Create report
    report = Report(metrics=[
        RegressionQualityMetric(),
        RegressionPredictedVsActualScatter(),
        RegressionErrorDistribution(),
        RegressionAbsPercentageErrorPlot()
    ])
    
    # Generate report
    report.run(
        reference_data=reference_with_pred,
        current_data=current_with_pred,
        column_mapping=column_mapping
    )
    
    return report


def save_report(report, filepath, title):
    """
    Save Evidently report as HTML.
    
    Args:
        report: Evidently report object
        filepath: Path to save HTML file
        title: Title for the report
    """
    print(f"\nSaving {title} to {filepath}...")
    
    # Save as HTML
    report.save_html(str(filepath))
    
    print(f"Report saved successfully")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 60)
    print("Model Drift Analysis")
    print("=" * 60)
    print(f"MLflow Tracking URI: {args.tracking_uri}")
    print(f"Model: {args.model_name}")
    print(f"Production data: Last {args.production_months} months")
    
    # Set MLflow tracking URI
    global MLFLOW_TRACKING_URI
    MLFLOW_TRACKING_URI = args.tracking_uri
    
    # Load data
    train_df, test_df = load_data()
    
    # Extract production data (last N months)
    production_df = extract_production_data(test_df, months=args.production_months)
    
    # Load model
    model = load_model_from_registry(args.model_name)
    
    # Generate predictions
    print("\nGenerating predictions...")
    train_pred = generate_predictions(model, train_df)
    production_pred = generate_predictions(model, production_df)
    
    print(f"Training predictions: {len(train_pred):,}")
    print(f"Production predictions: {len(production_pred):,}")
    
    # Create data drift report
    data_drift_report = create_data_drift_report(train_df, production_df)
    data_drift_path = REPORTS_DIR / f"data_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    save_report(data_drift_report, data_drift_path, "Data Drift Report")
    
    # Create performance report
    performance_report = create_performance_report(
        train_df, production_df,
        train_pred, production_pred
    )
    performance_path = REPORTS_DIR / f"performance_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    save_report(performance_report, performance_path, "Performance Drift Report")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Drift Analysis Summary")
    print("=" * 60)
    print(f"\nReference data (training):")
    print(f"  Rows: {len(train_df):,}")
    print(f"  Date range: {train_df.index.min()} to {train_df.index.max()}")
    print(f"\nProduction data (last {args.production_months} months):")
    print(f"  Rows: {len(production_df):,}")
    print(f"  Date range: {production_df.index.min()} to {production_df.index.max()}")
    print(f"\nReports saved:")
    print(f"  Data Drift: {data_drift_path}")
    print(f"  Performance Drift: {performance_path}")
    print("\n" + "=" * 60)
    print("Drift Analysis Completed!")
    print("=" * 60)
    print("\nOpen the HTML reports in your browser to view detailed drift analysis.")


if __name__ == "__main__":
    main()

