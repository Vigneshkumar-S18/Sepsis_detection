import os
import sys
import json
import datetime
import pandas as pd

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer, get_all_psv_files, validate_columns
from preprocessing.load_data import load_patients_parallel

# 41 original columns + 'PatientID' added at index 0
EXPECTED_COLUMNS = [
    'PatientID', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
    'SepsisLabel'
]

def merge_all_datasets():
    """
    Main function to coordinate finding, parallel loading, validating,
    and merging training_setA and training_setB into CSV and Parquet.
    """
    raw_dir_a = os.path.join(project_root, "datasets", "raw", "training_setA")
    raw_dir_b = os.path.join(project_root, "datasets", "raw", "training_setB")
    output_dir = os.path.join(project_root, "datasets", "interim")
    csv_file = os.path.join(output_dir, "merged_dataset.csv")
    parquet_file = os.path.join(output_dir, "merged_dataset.parquet")
    metadata_dir = os.path.join(project_root, "datasets", "metadata")
    metadata_file = os.path.join(metadata_dir, "dataset_summary.json")
    
    logger.info("Initializing Sepsis Dataset Merging Pipeline...")
    
    # 1. Discover all PSV files in raw directories
    files_a = get_all_psv_files(raw_dir_a)
    files_b = get_all_psv_files(raw_dir_b)
    all_files = files_a + files_b
    
    if not all_files:
        logger.error("No raw patient PSV files found. Please check datasets/raw/ directory structure.")
        sys.exit(1)
        
    logger.info(f"Total patient files to process: {len(all_files)}")
    
    # 2. Load all files in parallel
    merged_df = load_patients_parallel(all_files)
    
    if merged_df.empty:
        logger.error("Loading failed, merged DataFrame is empty.")
        sys.exit(1)
        
    # 3. Validate columns
    try:
        validate_columns(merged_df, EXPECTED_COLUMNS)
        logger.info("Dataset validation passed successfully.")
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        sys.exit(1)
        
    # 4. Save merged dataset
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    with Timer("Saving merged dataset to CSV (this may take a moment)"):
        merged_df.to_csv(csv_file, index=False)
    logger.info(f"CSV dataset saved to: {csv_file}")
    
    # Save Parquet
    with Timer("Saving merged dataset to Parquet"):
        merged_df.to_parquet(parquet_file, index=False, engine='pyarrow')
    logger.info(f"Parquet dataset saved to: {parquet_file}")
    
    # 5. Generate and Save Metadata Summary Report
    os.makedirs(metadata_dir, exist_ok=True)
    csv_size_mb = round(os.path.getsize(csv_file) / (1024 * 1024), 1)
    
    summary_data = {
        "patients": int(merged_df['PatientID'].nunique()),
        "records": int(len(merged_df)),
        "features": int(len(merged_df.columns) - 1),  # Exclude PatientID
        "created": datetime.date.today().strftime("%Y-%m-%d"),
        "dataset": "PhysioNet Challenge 2019",
        "size_mb": csv_size_mb
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Metadata summary saved to: {metadata_file}")
    
    # Log quick dataset summary
    logger.info(f"Summary: {summary_data['patients']} unique patients, {summary_data['records']} total records.")

if __name__ == "__main__":
    merge_all_datasets()

