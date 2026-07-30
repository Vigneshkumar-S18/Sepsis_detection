import os
import sys
import glob
import time
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer

def read_psv_pyarrow(f, opts):
    try:
        return pv.read_csv(f, parse_options=opts)
    except Exception as e:
        logger.warning(f"Error reading {f}: {e}")
        return None

def prepare_new_dataset_pipeline():
    """
    Ingests all 10,616 PSV patient files from newpreprocessedtrainingdata/completed_preprocessed_complete/
    using PyArrow C++ fast parser, standardizes column names, cleans NaNs, performs patient-stratified
    Train/Val/Test splits (70/15/15), fits a StandardScaler on Train features, and saves the parquet splits to datasets/processed/.
    """
    data_dir = os.path.join(project_root, "newpreprocessedtrainingdata", "completed_preprocessed_complete")
    psv_files = glob.glob(os.path.join(data_dir, "*.psv"))
    psv_files.sort()
    
    logger.info(f"Discovered {len(psv_files):,} patient files in {data_dir}")
    if not psv_files:
        logger.error(f"No PSV files found in {data_dir}!")
        sys.exit(1)
        
    opts = pv.ParseOptions(delimiter='|')
    
    with Timer(f"Fast reading {len(psv_files):,} patient files with PyArrow C++ parser"):
        dfs = []
        for i, f in enumerate(psv_files):
            if os.path.getsize(f) == 0:
                continue
            tbl = read_psv_pyarrow(f, opts)
            if tbl is not None and tbl.num_rows > 0:
                dfs.append(tbl.to_pandas())
            if (i + 1) % 2500 == 0:
                logger.info(f"  Parsed {i+1}/{len(psv_files)} files...")
                
        logger.info(f"Concatenating {len(dfs):,} patient DataFrames...")
        master_df = pd.concat(dfs, ignore_index=True)
        
    logger.info(f"Loaded master dataset: {len(master_df):,} rows, {len(master_df.columns)} columns.")
    
    # 1. Standardize Patient ID column name
    if 'Patient_ID' in master_df.columns and 'PatientID' not in master_df.columns:
        master_df.rename(columns={'Patient_ID': 'PatientID'}, inplace=True)
        
    if 'PatientID' not in master_df.columns:
        logger.error("PatientID column not found in dataset!")
        sys.exit(1)
        
    # Move PatientID to first column and SepsisLabel to second column
    cols = list(master_df.columns)
    cols.remove('PatientID')
    cols.remove('SepsisLabel')
    master_df = master_df[['PatientID', 'SepsisLabel'] + cols]
    
    # 2. Handle missing values (fill NaNs in columns like EtCO2 with 0.0)
    nan_cols = master_df.columns[master_df.isna().any()].tolist()
    if nan_cols:
        logger.info(f"Filling NaNs in {len(nan_cols)} columns with 0.0: {nan_cols[:10]}...")
        master_df.fillna(0.0, inplace=True)
        
    logger.info(f"Verification: remaining NaNs in dataset: {master_df.isna().sum().sum()}")
    
    # 3. Patient-wise stratified splitting (70% Train, 15% Val, 15% Test)
    with Timer("Patient-wise stratified train/val/test splitting"):
        patient_outcomes = master_df.groupby('PatientID')['SepsisLabel'].max().reset_index()
        patient_ids = patient_outcomes['PatientID'].values
        labels = patient_outcomes['SepsisLabel'].values
        
        train_patients, temp_patients, train_labels, temp_labels = train_test_split(
            patient_ids, labels, test_size=0.30, stratify=labels, random_state=42
        )
        
        val_patients, test_patients, val_labels, test_labels = train_test_split(
            temp_patients, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
        )
        
        train_set, val_set, test_set = set(train_patients), set(val_patients), set(test_patients)
        
        train_df = master_df[master_df['PatientID'].isin(train_set)].copy()
        val_df = master_df[master_df['PatientID'].isin(val_set)].copy()
        test_df = master_df[master_df['PatientID'].isin(test_set)].copy()
        
    logger.info("Split Summary:")
    logger.info(f"  Train: {len(train_patients):,} patients, {len(train_df):,} records ({train_labels.mean()*100:.2f}% septic)")
    logger.info(f"  Val:   {len(val_patients):,} patients, {len(val_df):,} records ({val_labels.mean()*100:.2f}% septic)")
    logger.info(f"  Test:  {len(test_patients):,} patients, {len(test_df):,} records ({test_labels.mean()*100:.2f}% septic)")
    
    # 4. Standard Feature Scaling
    with Timer("Fitting StandardScaler on Training features"):
        feature_cols = [c for c in train_df.columns if c not in ['PatientID', 'SepsisLabel']]
        
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols])
        
        train_df[feature_cols] = scaler.transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
    # 5. Save output datasets and scaler
    processed_dir = os.path.join(project_root, "datasets", "processed")
    interim_dir = os.path.join(project_root, "datasets", "interim")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(interim_dir, exist_ok=True)
    
    with Timer("Saving parquet splits and scaler.pkl"):
        master_df.to_parquet(os.path.join(interim_dir, "merged_dataset.parquet"), index=False)
        
        # Save feature parquets expected by downstream pipeline
        train_df.to_parquet(os.path.join(processed_dir, "train_features.parquet"), index=False)
        val_df.to_parquet(os.path.join(processed_dir, "validation_features.parquet"), index=False)
        test_df.to_parquet(os.path.join(processed_dir, "test_features.parquet"), index=False)
        
        train_df.to_parquet(os.path.join(processed_dir, "train_processed.parquet"), index=False)
        val_df.to_parquet(os.path.join(processed_dir, "validation_processed.parquet"), index=False)
        test_df.to_parquet(os.path.join(processed_dir, "test_processed.parquet"), index=False)
        
        scaler_path = os.path.join(processed_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
    logger.info(f"Dataset preparation complete! Total features: {len(feature_cols)}. All files saved to {processed_dir}")

if __name__ == "__main__":
    prepare_new_dataset_pipeline()
