import os
import sys
import json
import datetime
import pandas as pd

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer, get_paths

def run_save_processed(train_df, val_df, test_df):
    """
    Saves final preprocessed datasets in Parquet format to datasets/processed/
    and saves the preprocessing_metadata.json documentation file.
    """
    logger.info("Saving processed datasets and pipeline metadata...")
    processed_dir = os.path.join(project_root, "datasets", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Parquet files saving
    train_path = os.path.join(processed_dir, "train_processed.parquet")
    val_path = os.path.join(processed_dir, "validation_processed.parquet")
    test_path = os.path.join(processed_dir, "test_processed.parquet")
    
    with Timer("Writing train_processed.parquet"):
        train_df.to_parquet(train_path, index=False)
    logger.info(f"Saved training dataset: {train_path}")
    
    with Timer("Writing validation_processed.parquet"):
        val_df.to_parquet(val_path, index=False)
    logger.info(f"Saved validation dataset: {val_path}")
    
    with Timer("Writing test_processed.parquet"):
        test_df.to_parquet(test_path, index=False)
    logger.info(f"Saved testing dataset: {test_path}")
    
    # 2. Compile pipeline metadata
    # Count clinical features (excluding identifiers and target labels)
    num_features = len([c for c in train_df.columns if c not in ["PatientID", "SepsisLabel"]])
    
    metadata = {
        "imputation": "Forward Fill (6-hour limit vitals) + Full Lab Forward Fill + Train Medians",
        "scaler": "StandardScaler (parameters fit on Training split exclusively)",
        "patient_split": "70% Train, 15% Validation, 15% Test (Patient-wise Stratified)",
        "features": int(num_features),
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "dataset_shapes": {
            "train_split": train_df.shape,
            "validation_split": val_df.shape,
            "test_split": test_df.shape
        }
    }
    
    metadata_path = os.path.join(processed_dir, "preprocessing_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved pipeline metadata JSON to: {metadata_path}")
    
    return metadata

if __name__ == "__main__":
    with Timer("Step 3.8 - Save Final Processed Dataset"):
        processed_dir = os.path.join(project_root, "datasets", "processed")
        train_path = os.path.join(processed_dir, "train_split.parquet")
        val_path = os.path.join(processed_dir, "val_split.parquet")
        test_path = os.path.join(processed_dir, "test_split.parquet")
        
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            logger.error("Split parquets not found. Please run split_data.py first.")
            sys.exit(1)
            
        train = pd.read_parquet(train_path)
        val = pd.read_parquet(val_path)
        test = pd.read_parquet(test_path)
        
        run_save_processed(train, val, test)
