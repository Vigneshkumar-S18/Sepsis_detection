import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer

# Laboratory variables to create missingness indicators for
LAB_COLUMNS = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets'
]

def split_and_prepare_dataset():
    """
    Loads the merged dataset, generates binary missingness indicators for labs,
    performs patient-wise stratified splitting (70% Train, 15% Val, 15% Test),
    and saves the splits in datasets/processed/.
    """
    interim_dir = os.path.join(project_root, "datasets", "interim")
    parquet_path = os.path.join(interim_dir, "merged_dataset.parquet")
    output_dir = os.path.join(project_root, "datasets", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(parquet_path):
        logger.error(f"Merged dataset Parquet not found at: {parquet_path}. Please run merge_patients.py first.")
        sys.exit(1)
        
    with Timer("Loading merged Parquet dataset"):
        df = pd.read_parquet(parquet_path)
        
    # 1. Add missingness indicators for labs
    with Timer("Creating missingness indicators"):
        for col in LAB_COLUMNS:
            if col in df.columns:
                df[f"{col}_measured"] = df[col].notnull().astype(int)
            else:
                logger.warning(f"Lab column '{col}' not found in dataset columns.")
                
    # 2. Patient-wise stratified splits
    with Timer("Patient-wise stratified train/val/test splitting"):
        # Determine ever-septic status for stratification
        patient_outcomes = df.groupby('PatientID')['SepsisLabel'].max().reset_index()
        patient_ids = patient_outcomes['PatientID'].values
        labels = patient_outcomes['SepsisLabel'].values
        
        # Split patients: 70% Train, 30% Temp (Val + Test)
        train_patients, temp_patients, train_labels, temp_labels = train_test_split(
            patient_ids, labels, test_size=0.30, stratify=labels, random_state=42
        )
        
        # Split Temp: 50% Val (15% of total), 50% Test (15% of total)
        val_patients, test_patients, val_labels, test_labels = train_test_split(
            temp_patients, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
        )
        
        # Convert to sets for O(1) lookup
        train_set = set(train_patients)
        val_set = set(val_patients)
        test_set = set(test_patients)
        
        # Map back to rows in the main DataFrame
        train_df = df[df['PatientID'].isin(train_set)].copy()
        val_df = df[df['PatientID'].isin(val_set)].copy()
        test_df = df[df['PatientID'].isin(test_set)].copy()
        
    # Log shapes and ratios
    logger.info(f"Split results summary:")
    logger.info(f"  Train: {len(train_patients):,} patients ({len(train_df):,} records, {train_labels.mean()*100:.2f}% septic)")
    logger.info(f"  Val:   {len(val_patients):,} patients ({len(val_df):,} records, {val_labels.mean()*100:.2f}% septic)")
    logger.info(f"  Test:  {len(test_patients):,} patients ({len(test_df):,} records, {test_labels.mean()*100:.2f}% septic)")
    
    # 3. Save splits in Parquet format to datasets/processed/
    with Timer("Saving splits to datasets/processed/"):
        train_df.to_parquet(os.path.join(output_dir, "train_split.parquet"), index=False)
        val_df.to_parquet(os.path.join(output_dir, "val_split.parquet"), index=False)
        test_df.to_parquet(os.path.join(output_dir, "test_split.parquet"), index=False)
        
    logger.info(f"Splits saved successfully to: {output_dir}")

if __name__ == "__main__":
    split_and_prepare_dataset()
