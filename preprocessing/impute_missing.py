import os
import sys
import json
import pandas as pd

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer, get_paths
from preprocessing.config import VITALS_COLUMNS, LAB_COLUMNS

def run_adaptive_imputation(train_df, val_df, test_df):
    """
    Imputes missing clinical vital signs and laboratory features.
    1. Applies patient-wise forward-fill with a 6-hour limit on vital signs.
    2. Applies patient-wise forward-fill on sparse laboratory variables (no limit).
    3. Calculates feature medians exclusively on the Training split.
    4. Imputes all remaining NaNs (e.g. at the start of a stay or never-measured labs)
       across all splits with the Training medians.
    """
    logger.info("Starting adaptive missing value imputation...")
    
    # 1. Patient-wise forward fill for Vitals (limit=6 hours)
    logger.info("Applying bounded forward-fill (6-hour limit) for vital signs per patient...")
    for df in [train_df, val_df, test_df]:
        df[VITALS_COLUMNS] = df.groupby('PatientID')[VITALS_COLUMNS].ffill(limit=6)
        
    # 2. Patient-wise forward fill for Laboratory parameters (unlimited carry-forward)
    logger.info("Applying unlimited forward-fill for laboratory variables per patient...")
    for df in [train_df, val_df, test_df]:
        df[LAB_COLUMNS] = df.groupby('PatientID')[LAB_COLUMNS].ffill()
        
    # 3. Calculate medians on the Training split only (to prevent data leakage!)
    logger.info("Calculating feature medians on Training split...")
    clinical_features = VITALS_COLUMNS + LAB_COLUMNS
    train_medians = train_df[clinical_features].median()
    
    # Convert series to dictionary for JSON serialization and logging
    train_medians_dict = train_medians.to_dict()
    
    # For any columns that are entirely null in the training set (if any, though rare),
    # use a fallback (e.g. 0 or vital normal values), but here we default to 0.0 just in case.
    for col in clinical_features:
        if pd.isnull(train_medians_dict[col]):
            train_medians_dict[col] = 0.0
            
    # 4. Fill remaining NaNs with the Training medians
    logger.info("Imputing remaining missing values with Training medians...")
    train_df[clinical_features] = train_df[clinical_features].fillna(train_medians_dict)
    val_df[clinical_features] = val_df[clinical_features].fillna(train_medians_dict)
    test_df[clinical_features] = test_df[clinical_features].fillna(train_medians_dict)
    
    # 5. Handle demographics / administrative missingness
    logger.info("Imputing remaining demographic and administrative columns...")
    for df in [train_df, val_df, test_df]:
        if 'Unit1' in df.columns:
            df['Unit1'] = df['Unit1'].fillna(0.0)
        if 'Unit2' in df.columns:
            df['Unit2'] = df['Unit2'].fillna(0.0)
            
    train_adm_median = train_df['HospAdmTime'].median() if 'HospAdmTime' in train_df.columns else 0.0
    if pd.isnull(train_adm_median):
        train_adm_median = 0.0
    for df in [train_df, val_df, test_df]:
        if 'HospAdmTime' in df.columns:
            df['HospAdmTime'] = df['HospAdmTime'].fillna(train_adm_median)
            
    # Write summary reports
    reports_dir = os.path.join(project_root, "reports", "preprocessing")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save the medians to JSON
    medians_path = os.path.join(reports_dir, "imputation_summary.json")
    summary_data = {
        "imputation_strategy": {
            "vitals": "Forward fill (6h limit) + Train Medians",
            "labs": "Forward fill (unlimited) + Train Medians"
        },
        "train_medians": train_medians_dict
    }
    with open(medians_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved imputation summary to: {medians_path}")
    
    # Save missing value check after imputation to CSV
    missing_counts = {
        "Feature": clinical_features,
        "Train_Nulls": train_df[clinical_features].isnull().sum().values,
        "Val_Nulls": val_df[clinical_features].isnull().sum().values,
        "Test_Nulls": test_df[clinical_features].isnull().sum().values
    }
    missing_after_df = pd.DataFrame(missing_counts)
    missing_after_path = os.path.join(reports_dir, "missing_after.csv")
    missing_after_df.to_csv(missing_after_path, index=False)
    logger.info(f"Saved post-imputation check to: {missing_after_path}")
    
    return train_df, val_df, test_df, summary_data

if __name__ == "__main__":
    with Timer("Step 3.3 - Adaptive Imputation"):
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
        
        train, val, test, _ = run_adaptive_imputation(train, val, test)
        
        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)
        test.to_parquet(test_path, index=False)
        logger.info("Saved imputed datasets back to processed split files.")
