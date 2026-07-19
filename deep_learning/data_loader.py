# Sequence Data Loader for PyTorch Deep Learning Models
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SepsisSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for sepsis sequence data loading.
    Loads features (X) and labels (y) from generated npz archives.
    Supports optional sub-sampling to accelerate CPU-based runs.
    """
    def __init__(self, npz_path, sample_pct=1.0, seed=42):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Sequence archive not found: {npz_path}")
        
        # Load compressed data
        data = np.load(npz_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        pids = data['patient_ids']
        data.close()

        if sample_pct < 1.0:
            rng = np.random.RandomState(seed)
            n_samples = len(y)
            keep_size = max(100, int(n_samples * sample_pct))
            indices = rng.choice(n_samples, size=keep_size, replace=False)
            X = X[indices]
            y = y[indices]
            pids = pids[indices]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.patient_ids = pids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_sequence_dataloaders(config_dir, batch_size, sample_pct=1.0, logger=None):
    """
    Creates train, validation, and test PyTorch DataLoaders for a specific sequence config directory.

    Parameters
    ----------
    config_dir : str
        Path to the configuration folder (e.g. datasets/sequences/w12_h0/).
    batch_size : int
        Size of mini-batches.
    sample_pct : float
        Fraction of data to keep for acceleration.
    logger : logging.Logger, optional

    Returns
    -------
    tuple (train_loader, val_loader, test_loader)
    """
    train_path = os.path.join(config_dir, "train_sequences.npz")
    val_path = os.path.join(config_dir, "validation_sequences.npz")
    test_path = os.path.join(config_dir, "test_sequences.npz")

    if logger:
        logger.info(f"Loading sequence dataset splits from: {config_dir} (sample_pct: {sample_pct})")

    train_ds = SepsisSequenceDataset(train_path, sample_pct=sample_pct)
    val_ds = SepsisSequenceDataset(val_path, sample_pct=sample_pct)
    test_ds = SepsisSequenceDataset(test_path, sample_pct=1.0)  # Keep test set complete for fair evaluation

    # Shuffling training data is critical to optimize training descent
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False)

    if logger:
        logger.info(f"  Train sequences: {len(train_ds):,}")
        logger.info(f"  Val sequences:   {len(val_ds):,}")
        logger.info(f"  Test sequences:  {len(test_ds):,}")

    return train_loader, val_loader, test_loader
