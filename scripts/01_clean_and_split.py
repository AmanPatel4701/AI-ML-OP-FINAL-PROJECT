"""
Data Cleaning and Time-Based Splitting Script

This script:
1. Loads the raw household power consumption dataset
2. Cleans missing values, outliers, and data entry errors
3. Creates time-based features
4. Splits data chronologically into train (35%), validation (35%), and test (30%) sets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(file_path):
    """
    Load the raw semicolon-separated dataset.
    
    Args:
        file_path: Path to the raw data file
        
    Returns:
        DataFrame with raw data
    """
    print(f"Loading data from {file_path}...")
    
    # Read the semicolon-separated file
    # na_values=['?'] converts '?' to NaN
    df = pd.read_csv(
        file_path,
        sep=';',
        na_values=['?'],
        low_memory=False
    )
    
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df


def parse_datetime(df):
    """
    Combine Date and Time columns into a proper datetime index.
    
    Args:
        df: DataFrame with Date and Time columns
        
    Returns:
        DataFrame with datetime index
    """
    print("\nParsing datetime...")
    
    # Combine Date and Time columns
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    
    # Drop rows where datetime parsing failed
    initial_len = len(df)
    df = df.dropna(subset=['datetime'])
    dropped = initial_len - len(df)
    
    if dropped > 0:
        print(f"Dropped {dropped} rows with invalid datetime")
    
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Drop original Date and Time columns
    df = df.drop(columns=['Date', 'Time'])
    
    print(f"Datetime range: {df.index.min()} to {df.index.max()}")
    
    return df


def convert_to_numeric(df):
    """
    Convert all columns (except datetime index) to float.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with numeric columns
    """
    print("\nConverting columns to numeric...")
    
    # Convert all columns to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("All columns converted to numeric")
    
    return df


def handle_missing_values(df):
    """
    Handle missing values using forward-fill and median imputation.
    
    Args:
        df: DataFrame with missing values
        
    Returns:
        DataFrame with imputed values
    """
    print("\nHandling missing values...")
    
    # Count missing values before imputation
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before imputation: {missing_before:,}")
    
    # Forward-fill for time series data (carries last known value forward)
    df = df.ffill()
    
    # For any remaining missing values (e.g., at the start), use median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {df[col].isnull().sum()} missing values in {col} with median: {median_val:.2f}")
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after imputation: {missing_after:,}")
    
    return df


def remove_outliers(df):
    """
    Remove obvious outliers based on domain knowledge.
    
    Args:
        df: DataFrame with potential outliers
        
    Returns:
        DataFrame with outliers removed
    """
    print("\nRemoving outliers...")
    
    initial_len = len(df)
    
    # Remove voltage outliers (typical household voltage: 100-300V)
    voltage_outliers = (df['Voltage'] < 100) | (df['Voltage'] > 300)
    df = df[~voltage_outliers]
    print(f"  Removed {voltage_outliers.sum()} rows with voltage outliers")
    
    # Remove power outliers (typical household: < 15kW)
    power_outliers = df['Global_active_power'] > 15.0
    df = df[~power_outliers]
    print(f"  Removed {power_outliers.sum()} rows with power outliers (>15kW)")
    
    # Remove negative values (not physically possible for these measurements)
    negative_power = df['Global_active_power'] < 0
    df = df[~negative_power]
    print(f"  Removed {negative_power.sum()} rows with negative power")
    
    final_len = len(df)
    removed = initial_len - final_len
    print(f"Total rows removed: {removed:,} ({removed/initial_len*100:.2f}%)")
    print(f"Remaining rows: {final_len:,}")
    
    return df


def create_features(df):
    """
    Create time-based and lag features for better model performance.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with additional features
    """
    print("\nCreating features...")
    
    # Time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Lag features (1 hour ago)
    df['lag_1h'] = df['Global_active_power'].shift(1)
    
    # Rolling window features (3-hour rolling mean)
    df['rolling_mean_3h'] = df['Global_active_power'].rolling(window=3, min_periods=1).mean()
    
    # Drop rows where lag features are NaN (first few rows)
    df = df.dropna(subset=['lag_1h'])
    
    print(f"Created features: hour, dayofweek, month, is_weekend, lag_1h, rolling_mean_3h")
    print(f"Final feature count: {len(df.columns)}")
    
    return df


def split_chronologically(df):
    """
    Split data chronologically into train (35%), validation (35%), and test (30%).
    
    Args:
        df: DataFrame sorted by datetime
        
    Returns:
        train_df, validate_df, test_df
    """
    print("\nSplitting data chronologically...")
    
    # Ensure data is sorted by datetime
    df = df.sort_index()
    
    total_len = len(df)
    
    # Calculate split indices
    train_end = int(total_len * 0.35)
    validate_end = int(total_len * 0.70)  # 35% + 35% = 70%
    
    # Split the data
    train_df = df.iloc[:train_end].copy()
    validate_df = df.iloc[train_end:validate_end].copy()
    test_df = df.iloc[validate_end:].copy()
    
    print(f"Train set: {len(train_df):,} rows ({len(train_df)/total_len*100:.1f}%)")
    print(f"  Date range: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Validation set: {len(validate_df):,} rows ({len(validate_df)/total_len*100:.1f}%)")
    print(f"  Date range: {validate_df.index.min()} to {validate_df.index.max()}")
    print(f"Test set: {len(test_df):,} rows ({len(test_df)/total_len*100:.1f}%)")
    print(f"  Date range: {test_df.index.min()} to {test_df.index.max()}")
    
    return train_df, validate_df, test_df


def save_splits(train_df, validate_df, test_df, output_dir):
    """
    Save the three splits as CSV files.
    
    Args:
        train_df: Training DataFrame
        validate_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save CSV files
    """
    print("\nSaving splits to CSV...")
    
    train_path = output_dir / "train.csv"
    validate_path = output_dir / "validate.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=True)
    print(f"  Saved train.csv: {train_path}")
    
    validate_df.to_csv(validate_path, index=True)
    print(f"  Saved validate.csv: {validate_path}")
    
    test_df.to_csv(test_path, index=True)
    print(f"  Saved test.csv: {test_path}")
    
    print("\nData cleaning and splitting completed successfully!")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Household Electricity Data Cleaning and Splitting")
    print("=" * 60)
    
    # Step 1: Load raw data
    df = load_raw_data(RAW_DATA_PATH)
    
    # Step 2: Parse datetime
    df = parse_datetime(df)
    
    # Step 3: Convert to numeric
    df = convert_to_numeric(df)
    
    # Step 4: Handle missing values
    df = handle_missing_values(df)
    
    # Step 5: Remove outliers
    df = remove_outliers(df)
    
    # Step 6: Create features
    df = create_features(df)
    
    # Step 7: Split chronologically
    train_df, validate_df, test_df = split_chronologically(df)
    
    # Step 8: Save splits
    save_splits(train_df, validate_df, test_df, PROCESSED_DATA_DIR)
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print("\nTrain set target (Global_active_power) statistics:")
    print(train_df['Global_active_power'].describe())
    print("\nValidation set target statistics:")
    print(validate_df['Global_active_power'].describe())
    print("\nTest set target statistics:")
    print(test_df['Global_active_power'].describe())


if __name__ == "__main__":
    main()

