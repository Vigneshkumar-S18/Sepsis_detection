import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
import logging
from contextlib import contextmanager

# Add the project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def setup_logging(name="sepsis_eda"):
    """
    Sets up a standard, clean console logger for EDA.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logging()

@contextmanager
def Timer(operation_name):
    """
    Context manager to time execution of operations.
    """
    start_time = time.time()
    logger.info(f"Starting operation: {operation_name}...")
    yield
    elapsed = time.time() - start_time
    logger.info(f"Finished operation: {operation_name}. Elapsed time: {elapsed:.2f} seconds.")


def get_paths():
    """
    Returns resolved absolute paths to directories in the project.
    """
    paths = {
        "root": project_root,
        "raw": os.path.join(project_root, "datasets", "raw"),
        "interim": os.path.join(project_root, "datasets", "interim"),
        "metadata_dir": os.path.join(project_root, "datasets", "metadata"),
        "figures": os.path.join(project_root, "reports", "figures"),
        "tables": os.path.join(project_root, "reports", "tables"),
        "statistics": os.path.join(project_root, "reports", "statistics"),
        "summary": os.path.join(project_root, "reports", "summary")
    }
    
    # Ensure they exist (except raw/interim which are expected to exist)
    for k in ["figures", "tables", "statistics", "summary"]:
        os.makedirs(paths[k], exist_ok=True)
        
    return paths

def load_dataset(format="parquet"):
    """
    Loads the merged dataset from interim folder (defaults to Parquet for speed).
    """
    paths = get_paths()
    if format == "parquet":
        filepath = os.path.join(paths["interim"], "merged_dataset.parquet")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Parquet file not found: {filepath}. Run merge_patients.py first.")
        return pd.read_parquet(filepath)
    else:
        filepath = os.path.join(paths["interim"], "merged_dataset.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}. Run merge_patients.py first.")
        return pd.read_csv(filepath)

def set_plot_style():
    """
    Applies consistent styling configuration for publication-ready figures.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Palette definition (sleek modern medical tones: teal, coral, slate)
    palette = {
        "primary": "#0f766e",    # Deep Teal
        "secondary": "#f43f5e",  # Coral/Rose
        "neutral_dark": "#334155", # Slate
        "neutral_light": "#f8fafc", # Ice White
        "accent": "#0284c7"       # Soft Sky Blue
    }
    return palette
