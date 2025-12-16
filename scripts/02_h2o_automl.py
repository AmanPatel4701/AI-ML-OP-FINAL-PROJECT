"""
H2O AutoML Analysis Script

This script:
1. Loads the cleaned training dataset
2. Runs H2O AutoML to identify top-performing models
3. Displays the leaderboard
4. Identifies the top 3 model types for manual training
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

# H2O AutoML parameters
MAX_MODELS = 20
SEED = 42
MAX_RUNTIME_SECS = 1800  # 30 minutes
TARGET_COLUMN = "Global_active_power"


def initialize_h2o():
    """Initialize H2O cluster."""
    print("Initializing H2O cluster...")
    h2o.init(nthreads=-1, max_mem_size="4G")
    print("H2O cluster initialized successfully")


def load_training_data(file_path):
    """
    Load the training dataset.
    
    Args:
        file_path: Path to train.csv
        
    Returns:
        H2OFrame with training data
    """
    print(f"\nLoading training data from {file_path}...")
    
    # Load CSV into H2O
    hf = h2o.import_file(str(file_path))
    
    print(f"Loaded {hf.nrows:,} rows and {hf.ncols} columns")
    print(f"Column names: {hf.names}")
    
    return hf


def prepare_data(hf, target_column):
    """
    Prepare data for AutoML by identifying features and target.
    
    Args:
        hf: H2OFrame
        target_column: Name of the target column
        
    Returns:
        features: List of feature column names
        target: Target column name
    """
    # Identify features (all columns except target and datetime index)
    all_columns = hf.names
    
    # Remove target and datetime index if present
    features = [col for col in all_columns if col != target_column and col != 'datetime']
    
    print(f"\nTarget variable: {target_column}")
    print(f"Features ({len(features)}): {features}")
    
    return features, target_column


def run_automl(hf, features, target, max_models, seed, max_runtime_secs):
    """
    Run H2O AutoML.
    
    Args:
        hf: H2OFrame with training data
        features: List of feature column names
        target: Target column name
        max_models: Maximum number of models to train
        seed: Random seed for reproducibility
        max_runtime_secs: Maximum runtime in seconds
        
    Returns:
        aml: H2OAutoML object
    """
    print("\n" + "=" * 60)
    print("Running H2O AutoML...")
    print("=" * 60)
    print(f"Max models: {max_models}")
    print(f"Max runtime: {max_runtime_secs} seconds ({max_runtime_secs/60:.1f} minutes)")
    print(f"Seed: {seed}")
    print("\nThis may take a while. Please wait...")
    
    # Initialize AutoML
    aml = H2OAutoML(
        max_models=max_models,
        seed=seed,
        max_runtime_secs=max_runtime_secs,
        sort_metric="RMSE",  # For regression, sort by RMSE
        stopping_metric="RMSE",
        stopping_tolerance=0.001,
        stopping_rounds=3,
        balance_classes=False,  # Not needed for regression
        nfolds=5,  # 5-fold cross-validation
        keep_cross_validation_predictions=True,
        keep_cross_validation_models=True,
        verbosity="info"
    )
    
    # Train AutoML
    aml.train(
        x=features,
        y=target,
        training_frame=hf
    )
    
    print("\nAutoML training completed!")
    
    return aml


def display_leaderboard(aml):
    """
    Display the AutoML leaderboard.
    
    Args:
        aml: H2OAutoML object
    """
    print("\n" + "=" * 60)
    print("H2O AutoML Leaderboard")
    print("=" * 60)
    
    # Get leaderboard
    lb = aml.leaderboard
    
    # Display leaderboard
    print(lb.as_data_frame())
    
    # Save leaderboard to CSV
    leaderboard_path = Path(__file__).parent.parent / "reports" / "h2o_leaderboard.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    lb_df = lb.as_data_frame()
    lb_df.to_csv(leaderboard_path, index=False)
    print(f"\nLeaderboard saved to: {leaderboard_path}")


def identify_top_3_models(aml):
    """
    Identify the top 3 model types from the leaderboard.
    
    Args:
        aml: H2OAutoML object
        
    Returns:
        top_3_models: List of top 3 model identifiers
    """
    print("\n" + "=" * 60)
    print("Top 3 Model Types Identification")
    print("=" * 60)
    
    # Get leaderboard
    lb = aml.leaderboard
    lb_df = lb.as_data_frame()
    
    # Extract model types from model_id
    # H2O model IDs typically contain the algorithm name
    top_3_models = []
    model_types_seen = set()
    
    print("\nAnalyzing top models...")
    for idx, row in lb_df.head(10).iterrows():  # Check top 10 to find 3 unique types
        model_id = row['model_id']
        
        # Extract model type from model_id
        # Common patterns: GBM, XGBoost, GLM, DeepLearning, StackedEnsemble, etc.
        model_type = None
        
        if 'XGBoost' in model_id or 'XGBoost' in model_id:
            model_type = 'XGBoost'
        elif 'GBM' in model_id and 'StackedEnsemble' not in model_id:
            model_type = 'GBM'
        elif 'GLM' in model_id:
            model_type = 'GLM'
        elif 'DeepLearning' in model_id or 'Deep' in model_id:
            model_type = 'DeepLearning'
        elif 'StackedEnsemble' in model_id:
            # Skip stacked ensembles for now, prefer base models
            continue
        elif 'DRF' in model_id:
            model_type = 'DRF'
        else:
            # Try to extract from model_id
            parts = model_id.split('_')
            for part in parts:
                if part in ['GBM', 'GLM', 'XGBoost', 'DeepLearning', 'DRF']:
                    model_type = part
                    break
        
        if model_type and model_type not in model_types_seen:
            model_types_seen.add(model_type)
            top_3_models.append({
                'rank': len(top_3_models) + 1,
                'model_type': model_type,
                'model_id': model_id,
                'rmse': row['rmse'],
                'mae': row.get('mae', 'N/A'),
                'r2': row.get('r2', 'N/A')
            })
            
            print(f"\nRank {len(top_3_models)}: {model_type}")
            print(f"  Model ID: {model_id}")
            print(f"  RMSE: {row['rmse']:.4f}")
            if 'mae' in row:
                print(f"  MAE: {row['mae']:.4f}")
            if 'r2' in row:
                print(f"  R²: {row['r2']:.4f}")
        
        if len(top_3_models) >= 3:
            break
    
    # If we don't have 3 unique types, add common ones
    if len(top_3_models) < 3:
        common_models = ['XGBoost', 'GBM', 'GLM']
        for model_type in common_models:
            if model_type not in model_types_seen:
                top_3_models.append({
                    'rank': len(top_3_models) + 1,
                    'model_type': model_type,
                    'model_id': 'N/A',
                    'rmse': 'N/A',
                    'mae': 'N/A',
                    'r2': 'N/A'
                })
                print(f"\nRank {len(top_3_models)}: {model_type} (default selection)")
            if len(top_3_models) >= 3:
                break
    
    print("\n" + "=" * 60)
    print("RECOMMENDED TOP 3 MODEL TYPES FOR MANUAL TRAINING:")
    print("=" * 60)
    for model in top_3_models:
        print(f"{model['rank']}. {model['model_type']}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("Train these three model types manually using:")
    print("  - scripts/train_xgboost.py")
    print("  - scripts/train_gbm.py")
    print("  - scripts/train_glm.py")
    
    return top_3_models


def main():
    """Main execution function."""
    print("=" * 60)
    print("H2O AutoML Analysis")
    print("=" * 60)
    
    try:
        # Step 1: Initialize H2O
        initialize_h2o()
        
        # Step 2: Load training data
        hf = load_training_data(TRAIN_DATA_PATH)
        
        # Step 3: Prepare data
        features, target = prepare_data(hf, TARGET_COLUMN)
        
        # Step 4: Run AutoML
        aml = run_automl(
            hf, features, target,
            MAX_MODELS, SEED, MAX_RUNTIME_SECS
        )
        
        # Step 5: Display leaderboard
        display_leaderboard(aml)
        
        # Step 6: Identify top 3 models
        top_3_models = identify_top_3_models(aml)
        
        print("\n" + "=" * 60)
        print("AutoML Analysis Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
    finally:
        # Shutdown H2O
        print("\nShutting down H2O cluster...")
        h2o.cluster().shutdown()


if __name__ == "__main__":
    main()

