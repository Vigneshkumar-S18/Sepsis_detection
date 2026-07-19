# Data Loader for Classical ML Baselines
import os
import pandas as pd
from baseline_models.config import EXCLUDE_COLUMNS, LABEL_COLUMN


def load_dataset(dataset_type, project_root, split):
    """
    Loads train, validation, or test split for Dataset A or Dataset B.

    Parameters
    ----------
    dataset_type : str
        Either 'Dataset A' (original 68 features) or 'Dataset B' (engineered 97 features).
    project_root : str
        Path to the project root directory.
    split : str
        Either 'train', 'validation', or 'test'.

    Returns
    -------
    tuple (X, y)
        X : pd.DataFrame (features)
        y : pd.Series (binary targets)
    """
    processed_dir = os.path.join(project_root, "datasets", "processed")

    if dataset_type == "Dataset A":
        filename = f"{split}_processed.parquet"
    elif dataset_type == "Dataset B":
        filename = f"{split}_features.parquet"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    file_path = os.path.join(processed_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed split file not found: {file_path}")

    # Load parquet
    df = pd.read_parquet(file_path)

    # Separate features and target
    y = df[LABEL_COLUMN]
    X = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns])

    return X, y
