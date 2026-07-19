import os
import sys
import pandas as pd
import numpy as np

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer, get_paths
from preprocessing.config import CLINICAL_RANGES

def run_range_validation(train_df, val_df, test_df):
    """
    Validates clinical ranges for key vitals and pH across all splits.
    Replaces physiologically impossible out-of-bound values with NaN,
    and logs the summary of corrections.
    """
    reports_dir = os.path.join(project_root, "reports", "preprocessing")
    os.makedirs(reports_dir, exist_ok=True)
    
    validation_summary = []
    invalid_corrections = []
    
    # Process each clinical range rule
    for feature, (min_val, max_val) in CLINICAL_RANGES.items():
        logger.info(f"Validating range for {feature}: [{min_val}, {max_val}]")
        
        feature_stats = {
            "Feature": feature,
            "AcceptableRange": f"{min_val} - {max_val}",
            "Train_Invalid_Count": 0,
            "Val_Invalid_Count": 0,
            "Test_Invalid_Count": 0
        }
        
        # Helper to apply range validation to a single split
        for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if feature not in df.columns:
                continue
                
            col_data = df[feature]
            # Find out-of-bound non-null values
            invalid_mask = col_data.notnull() & ((col_data < min_val) | (col_data > max_val))
            invalid_indices = df[invalid_mask].index
            invalid_count = len(invalid_indices)
            
            feature_stats[f"{split_name}_Invalid_Count"] = invalid_count
            
            if invalid_count > 0:
                logger.info(f"  {split_name} split: found {invalid_count} out-of-bound values for {feature}.")
                # Log details of invalid values (sample or all up to a reasonable limit)
                for idx in invalid_indices[:50]:  # Cap at 50 per feature per split to keep file small
                    row = df.loc[idx]
                    invalid_corrections.append({
                        "Split": split_name,
                        "PatientID": row["PatientID"],
                        "ICULOS": row["ICULOS"],
                        "Feature": feature,
                        "OriginalValue": row[feature],
                        "CorrectedValue": np.nan
                    })
                
                # Replace invalid values with NaN
                df.loc[invalid_indices, feature] = np.nan
                
        validation_summary.append(feature_stats)
        
    # Save CSV reports
    summary_df = pd.DataFrame(validation_summary)
    summary_path = os.path.join(reports_dir, "range_validation.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved range validation summary table to: {summary_path}")
    
    corrections_df = pd.DataFrame(invalid_corrections)
    if not corrections_df.empty:
        corr_path = os.path.join(reports_dir, "invalid_values.csv")
        corrections_df.to_csv(corr_path, index=False)
        logger.info(f"Saved invalid corrections detailed log to: {corr_path}")
    else:
        # Create empty placeholder if no invalid values found
        corr_path = os.path.join(reports_dir, "invalid_values.csv")
        pd.DataFrame(columns=["Split", "PatientID", "ICULOS", "Feature", "OriginalValue", "CorrectedValue"]).to_csv(corr_path, index=False)
        logger.info("No out-of-bound values found.")
        
    return train_df, val_df, test_df, validation_summary

if __name__ == "__main__":
    with Timer("Step 3.2 - Clinical Range Validation"):
        # Standalone execution
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
        
        train, val, test, _ = run_range_validation(train, val, test)
        
        # Save back to processed directory
        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)
        test.to_parquet(test_path, index=False)
        logger.info("Saved range-validated data back to processed split files.")
