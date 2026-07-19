# ─────────────────────────────────────────────────────────────────────────────
# Sequence Builder — Converts patient tabular data into temporal sequences
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sequence_generation.label_generator import get_label
from sequence_generation.metadata import build_sequence_metadata_record
from sequence_generation.config import PATIENT_ID_COLUMN, LABEL_COLUMN, EXCLUDE_COLUMNS


def build_patient_sequences(patient_df, feature_cols, window_size, horizon, stride):
    """
    Generates sliding-window sequences for a single patient.

    Parameters
    ----------
    patient_df : pd.DataFrame
        All hourly records for one patient, sorted by ICULOS.
    feature_cols : list[str]
        Column names to include in the feature tensor.
    window_size : int
        Number of contiguous hours in each observation window.
    horizon : int
        How many hours ahead of the window end to look for the label.
    stride : int
        Step size for the sliding window.

    Returns
    -------
    tuple (X_list, y_list, meta_list)
        X_list  : list of np.ndarray, each shape (window_size, n_features)
        y_list  : list of int labels
        meta_list : list of metadata dicts
    """
    patient_id = patient_df[PATIENT_ID_COLUMN].iloc[0]
    features = patient_df[feature_cols].values.astype(np.float32)
    labels = patient_df[LABEL_COLUMN].values
    iculos = patient_df["ICULOS"].values

    n_rows = len(patient_df)
    X_list = []
    y_list = []
    meta_list = []

    # Slide the window across the patient's stay
    for start in range(0, n_rows - window_size + 1, stride):
        end = start + window_size - 1  # inclusive end index

        # Get label at the prediction point
        label = get_label(labels, end, horizon)
        if label is None:
            # The prediction point falls beyond the patient's recorded stay
            continue

        # Extract the feature window
        X_window = features[start : start + window_size]  # shape (W, F)
        X_list.append(X_window)
        y_list.append(label)

        # Build metadata record using 1-based patient-relative hour indices
        # (not scaled ICULOS values, which are meaningless after standardization)
        meta = build_sequence_metadata_record(
            patient_id=patient_id,
            start_hour=start + 1,           # 1-based hour within this patient's stay
            end_hour=start + window_size,    # 1-based inclusive end hour
            prediction_hour=start + window_size + horizon,  # prediction target hour
            window_size=window_size,
            horizon=horizon,
            label=label,
            split_name="",  # filled in by the caller
        )
        meta_list.append(meta)

    return X_list, y_list, meta_list


def build_split_sequences(split_df, feature_cols, window_size, horizon, stride,
                          split_name, logger=None):
    """
    Generates sequences for an entire data split (train / validation / test).

    Groups by PatientID, calls build_patient_sequences per patient, and
    concatenates results into final numpy arrays.

    Parameters
    ----------
    split_df : pd.DataFrame
        The full split DataFrame (e.g. train_features.parquet loaded).
    feature_cols : list[str]
        Columns to include in the feature tensor.
    window_size : int
        Observation window length.
    horizon : int
        Prediction horizon.
    stride : int
        Sliding window stride.
    split_name : str
        Name of the split (train, validation, test).
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    tuple (X, y, patient_ids, metadata_df)
        X           : np.ndarray, shape (N, window_size, n_features), float32
        y           : np.ndarray, shape (N,), int32
        patient_ids : np.ndarray, shape (N,), object (patient ID strings)
        metadata_df : pd.DataFrame with N rows
    """
    all_X = []
    all_y = []
    all_meta = []

    grouped = split_df.groupby(PATIENT_ID_COLUMN, sort=False)
    n_patients = len(grouped)

    for i, (pid, patient_df) in enumerate(grouped):
        # Sort each patient by ICULOS to ensure temporal order
        patient_df = patient_df.sort_values("ICULOS")

        # Skip patients whose stay is shorter than the window + horizon
        if len(patient_df) < window_size:
            continue

        X_list, y_list, meta_list = build_patient_sequences(
            patient_df, feature_cols, window_size, horizon, stride
        )

        if len(X_list) > 0:
            # Stamp the split name into metadata
            for m in meta_list:
                m["Split"] = split_name
            all_X.extend(X_list)
            all_y.extend(y_list)
            all_meta.extend(meta_list)

        if logger and (i + 1) % 5000 == 0:
            logger.info(f"  Processed {i+1}/{n_patients} patients...")

    if len(all_X) == 0:
        if logger:
            logger.warning(f"No sequences generated for split={split_name}, "
                           f"window={window_size}, horizon={horizon}")
        empty_X = np.empty((0, window_size, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int32)
        empty_ids = np.empty((0,), dtype=object)
        empty_meta = pd.DataFrame()
        return empty_X, empty_y, empty_ids, empty_meta

    X = np.stack(all_X, axis=0)             # (N, W, F)
    y = np.array(all_y, dtype=np.int32)     # (N,)
    patient_ids = np.array([m["PatientID"] for m in all_meta], dtype=object)
    metadata_df = pd.DataFrame(all_meta)

    if logger:
        logger.info(f"  {split_name}: {X.shape[0]:,} sequences generated "
                     f"(shape: {X.shape})")

    return X, y, patient_ids, metadata_df
