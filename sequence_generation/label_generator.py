# ─────────────────────────────────────────────────────────────────────────────
# Label Generator — Extracts and validates labels for temporal sequences
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np


def get_label(sepsis_labels, window_end_idx, horizon):
    """
    Retrieves the SepsisLabel at a future time point relative to the window end.

    Parameters
    ----------
    sepsis_labels : np.ndarray
        Full array of SepsisLabel values for a single patient (sorted by ICULOS).
    window_end_idx : int
        Index of the last row in the current sliding window (0-indexed).
    horizon : int
        Number of hours ahead of window_end_idx to look for the label.

    Returns
    -------
    int or None
        The label (0 or 1) if the target index exists, otherwise None.
    """
    target_idx = window_end_idx + horizon
    if target_idx < len(sepsis_labels):
        return int(sepsis_labels[target_idx])
    return None


def compute_label_statistics(y_array):
    """
    Computes class distribution statistics for a label array.

    Parameters
    ----------
    y_array : np.ndarray
        1D array of binary labels.

    Returns
    -------
    dict
        Dictionary containing total, positive count, negative count,
        positive rate, and imbalance ratio.
    """
    total = len(y_array)
    positive = int(np.sum(y_array == 1))
    negative = total - positive
    positive_rate = positive / total if total > 0 else 0.0
    imbalance_ratio = negative / positive if positive > 0 else float('inf')

    return {
        "total_sequences": total,
        "positive_count": positive,
        "negative_count": negative,
        "positive_rate": round(positive_rate, 6),
        "imbalance_ratio": round(imbalance_ratio, 2),
    }
