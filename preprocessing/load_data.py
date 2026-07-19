import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from preprocessing.utils import logger, Timer

def load_single_patient(file_path):
    """
    Reads a single .psv file, extracts the PatientID from the filename,
    and inserts it as the first column of the DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep='|')
        # Extract filename without extension (e.g., 'p000001')
        patient_id = os.path.splitext(os.path.basename(file_path))[0]
        # Insert PatientID at index 0
        df.insert(0, 'PatientID', patient_id)
        return df
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None

def load_patients_parallel(file_paths, max_workers=None):
    """
    Loads all patient files in parallel using ThreadPoolExecutor
    and returns a single consolidated Pandas DataFrame.
    """
    if not file_paths:
        logger.warning("No file paths provided for loading.")
        return pd.DataFrame()
        
    dfs = []
    total_files = len(file_paths)
    
    with Timer(f"Parallel load of {total_files} files"):
        # We use ThreadPoolExecutor as it avoids process spawn overhead on Windows
        # and runs efficiently for I/O bound small file operations.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for loading
            future_to_file = {executor.submit(load_single_patient, fp): fp for fp in file_paths}
            
            completed = 0
            for future in as_completed(future_to_file):
                df = future.result()
                if df is not None:
                    dfs.append(df)
                completed += 1
                if completed % 5000 == 0 or completed == total_files:
                    logger.info(f"Loaded {completed}/{total_files} files.")
                    
    if not dfs:
        logger.error("No dataframes could be loaded.")
        return pd.DataFrame()
        
    with Timer("Concatenating patient DataFrames"):
        merged_df = pd.concat(dfs, ignore_index=True)
        
    logger.info(f"Combined data shape: {merged_df.shape}")
    return merged_df
