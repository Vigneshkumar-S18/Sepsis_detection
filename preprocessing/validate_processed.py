import os
import sys
import json
import pandas as pd

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer, get_paths

def run_processed_validation(train_df, val_df, test_df):
    """
    Quality Assurance checks on processed datasets:
    1. Asserts zero missing values remaining in any split.
    2. Confirms there are zero duplicate rows.
    3. Asserts zero patient leakage (train, val, test patient sets are mutually exclusive).
    4. Validates matching column counts and schemas across all splits.
    Saves a quality report to reports/preprocessing/quality_report.json.
    """
    logger.info("Executing Quality Assurance Validation on processed data...")
    reports_dir = os.path.join(project_root, "reports", "preprocessing")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Missing values remaining
    train_nulls = int(train_df.isnull().sum().sum())
    val_nulls = int(val_df.isnull().sum().sum())
    test_nulls = int(test_df.isnull().sum().sum())
    
    # 2. Duplicate rows
    train_duplicates = int(train_df.duplicated().sum())
    val_duplicates = int(val_df.duplicated().sum())
    test_duplicates = int(test_df.duplicated().sum())
    
    # 3. Patient sets and intersection (Leakage Check)
    train_patients = set(train_df['PatientID'].unique())
    val_patients = set(val_df['PatientID'].unique())
    test_patients = set(test_df['PatientID'].unique())
    
    leakage_train_val = list(train_patients.intersection(val_patients))
    leakage_train_test = list(train_patients.intersection(test_patients))
    leakage_val_test = list(val_patients.intersection(test_patients))
    
    has_leakage = len(leakage_train_val) > 0 or len(leakage_train_test) > 0 or len(leakage_val_test) > 0
    
    # 4. Schema/Columns check
    columns_match = (list(train_df.columns) == list(val_df.columns)) and (list(train_df.columns) == list(test_df.columns))
    
    # Compile QA outcomes
    qa_passed = (train_nulls == 0 and val_nulls == 0 and test_nulls == 0 and
                 not has_leakage and columns_match)
                 
    qa_report = {
        "qa_passed": qa_passed,
        "null_values_count": {
            "train": train_nulls,
            "val": val_nulls,
            "test": test_nulls
        },
        "duplicate_rows_count": {
            "train": train_duplicates,
            "val": val_duplicates,
            "test": test_duplicates
        },
        "leakage_checks": {
            "has_leakage": has_leakage,
            "train_vs_val_overlap_count": len(leakage_train_val),
            "train_vs_test_overlap_count": len(leakage_train_test),
            "val_vs_test_overlap_count": len(leakage_val_test),
            "leakage_patients": {
                "train_val": leakage_train_val,
                "train_test": leakage_train_test,
                "val_test": leakage_val_test
            }
        },
        "schema_checks": {
            "columns_match_exactly": columns_match,
            "train_columns_count": len(train_df.columns),
            "val_columns_count": len(val_df.columns),
            "test_columns_count": len(test_df.columns)
        },
        "dataset_shapes": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df)
        }
    }
    
    # Save JSON quality report
    report_path = os.path.join(reports_dir, "quality_report.json")
    with open(report_path, 'w') as f:
        json.dump(qa_report, f, indent=2)
    logger.info(f"Saved QA quality report to: {report_path}")
    
    if qa_passed:
        logger.info("Quality Assurance checks PASSED! Data is ready for ML training.")
    else:
        logger.error("Quality Assurance checks FAILED! Please inspect quality_report.json.")
        
    return qa_passed, qa_report

if __name__ == "__main__":
    with Timer("Step 3.7 - Quality Assurance Validation"):
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
        
        run_processed_validation(train, val, test)
