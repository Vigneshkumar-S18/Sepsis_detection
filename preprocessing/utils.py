import os
import time
import logging
from contextlib import contextmanager

def setup_logging(name="sepsis_preprocessing"):
    """
    Sets up a standard, clean console logger.
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

def get_all_psv_files(dir_path):
    """
    Discovers all .psv files in the given directory.
    """
    if not os.path.exists(dir_path):
        logger.error(f"Directory path does not exist: {dir_path}")
        return []
    
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".psv")]
    # Ensure stable sorting
    files.sort()
    logger.info(f"Discovered {len(files)} patient files in {dir_path}")
    return files

def validate_columns(df, expected_columns):
    """
    Validates that all expected columns are present in the dataframe.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in data: {missing}")
    return True
