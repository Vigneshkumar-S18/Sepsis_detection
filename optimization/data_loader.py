# Data Loader utilities for hyperparameter optimization
import os
import sys
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from baseline_models.config import EXCLUDE_COLUMNS, LABEL_COLUMN


def load_tabular_splits():
    """
    Loads train, validation, and test splits of Dataset B for XGBoost tuning.
    """
    processed_dir = os.path.join(project_root, "datasets", "processed")
    
    splits = {}
    for split in ["train", "validation", "test"]:
        file_path = os.path.join(processed_dir, f"{split}_features.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet features file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        y = df[LABEL_COLUMN]
        X = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns])
        splits[split] = (X, y)
        
    return splits["train"], splits["validation"], splits["test"]


def load_sequence_splits(cfg_id="w12_h0"):
    """
    Loads sequence archives for BiLSTM tuning.
    """
    seq_dir = os.path.join(project_root, "datasets", "sequences", cfg_id)
    
    splits = {}
    for split in ["train", "validation", "test"]:
        file_path = os.path.join(seq_dir, f"{split}_sequences.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sequence npz file not found: {file_path}")
            
        data = np.load(file_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        pids = data['patient_ids']
        data.close()
        
        splits[split] = (X, y, pids)
        
    return splits["train"], splits["validation"], splits["test"]
