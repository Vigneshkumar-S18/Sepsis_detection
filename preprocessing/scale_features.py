import os
import sys
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer, get_paths

def run_feature_scaling(train_df, val_df, test_df):
    """
    Fits a StandardScaler on the continuous numerical features of the Train set
    and applies the transformation across Train, Val, and Test splits.
    Saves the scaler pickle file and a statistics sheet.
    """
    logger.info("Scaling continuous numerical features...")
    reports_dir = os.path.join(project_root, "reports", "preprocessing")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Identify columns to scale
    # Continuous vitals, labs, age, HospAdmTime, and ICULOS.
    # Exclude IDs, target, binary indicators, and missingness flags.
    columns_to_scale = [
        col for col in train_df.columns
        if col not in ['PatientID', 'SepsisLabel', 'Gender', 'Unit1', 'Unit2']
        and not col.endswith('_measured')
    ]
    
    logger.info(f"Identified {len(columns_to_scale)} columns to scale.")
    
    # 2. Fit StandardScaler on Train split only (prevent leakage!)
    scaler = StandardScaler()
    scaler.fit(train_df[columns_to_scale])
    
    # 3. Apply standard scaling to all splits
    train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
    val_df[columns_to_scale] = scaler.transform(val_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
    
    # 4. Save scaler pickle to datasets/processed/
    processed_dir = os.path.join(project_root, "datasets", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved fitted scaler object to: {scaler_path}")
    
    # 5. Save scaled statistics to reports/preprocessing/scaled_statistics.csv
    # This tracks the mean and variance used to ensure reproducibility
    stats_data = {
        "Feature": columns_to_scale,
        "Mean": scaler.mean_,
        "Scale": scaler.scale_,
        "Variance": scaler.var_
    }
    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(reports_dir, "scaled_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Saved scaled statistics sheet to: {stats_path}")
    
    return train_df, val_df, test_df, scaler

if __name__ == "__main__":
    with Timer("Step 3.6 - Feature Scaling"):
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
        
        train, val, test, _ = run_feature_scaling(train, val, test)
        
        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)
        test.to_parquet(test_path, index=False)
        logger.info("Saved scaled datasets back to processed split files.")
