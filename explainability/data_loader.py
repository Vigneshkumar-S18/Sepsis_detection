# Data Loader for Explainability Modules
import os
import sys
import numpy as np
import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from explainability.config import LABEL_COLUMN, EXCLUDE_COLUMNS


def load_tabular_data(split="test"):
    """
    Loads features (X) and target (y) for classical tabular explanations (SHAP).
    """
    processed_dir = os.path.join(project_root, "datasets", "processed")
    file_path = os.path.join(processed_dir, f"{split}_features.parquet")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed tabular features not found: {file_path}")
        
    df = pd.read_parquet(file_path)
    y = df[LABEL_COLUMN]
    X = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns])
    return X, y


def load_sequence_data(cfg_id="w12_h0", split="test"):
    """
    Loads raw sequence data (X, y, patient_ids) from .npz files.
    """
    seq_path = os.path.join(project_root, "datasets", "sequences", cfg_id, f"{split}_sequences.npz")
    
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequence archive not found: {seq_path}")
        
    data = np.load(seq_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    pids = data['patient_ids']
    data.close()
    
    return X, y, pids


def get_feature_names():
    """
    Gets the column names (features) in the order they exist in X.
    """
    X, _ = load_tabular_data("test")
    return list(X.columns)
