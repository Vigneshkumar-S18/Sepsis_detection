# ─────────────────────────────────────────────────────────────────────────────
# Save Sequences — Persists sequence tensors and metadata to disk
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd


def save_sequence_dataset(X, y, patient_ids, metadata_df, output_dir,
                          split_name, logger=None):
    """
    Saves a single split's sequence dataset to compressed .npz and metadata .csv.

    Parameters
    ----------
    X : np.ndarray, shape (N, W, F), float32
    y : np.ndarray, shape (N,), int32
    patient_ids : np.ndarray, shape (N,), object
    metadata_df : pd.DataFrame
    output_dir : str
        Directory path (e.g. datasets/sequences/w12_h0/)
    split_name : str
        Split name (train, validation, test)
    logger : logging.Logger, optional
    """
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, f"{split_name}_sequences.npz")
    meta_path = os.path.join(output_dir, f"{split_name}_metadata.csv")

    # Save compressed numpy archive
    np.savez_compressed(npz_path, X=X, y=y, patient_ids=patient_ids)

    # Save metadata CSV
    metadata_df.to_csv(meta_path, index=False)

    if logger:
        npz_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
        logger.info(f"  Saved {split_name}: {npz_path} ({npz_size_mb:.1f} MB), "
                     f"{meta_path}")
