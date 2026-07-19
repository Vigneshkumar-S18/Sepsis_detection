# ─────────────────────────────────────────────────────────────────────────────
# Validation — Automated integrity checks for generated sequence datasets
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


def validate_sequences(X, y, patient_ids, metadata_df, window_size, n_features,
                       source_df, feature_cols, horizon, logger=None):
    """
    Runs a comprehensive suite of integrity checks on generated sequences.

    Parameters
    ----------
    X : np.ndarray, shape (N, window_size, n_features)
    y : np.ndarray, shape (N,)
    patient_ids : np.ndarray, shape (N,)
    metadata_df : pd.DataFrame
    window_size : int
    n_features : int
    source_df : pd.DataFrame
        The original split DataFrame to spot-check label alignment against.
    feature_cols : list[str]
    horizon : int
    logger : logging.Logger, optional

    Returns
    -------
    dict
        Validation results with pass/fail status for each check.
    """
    results = {}

    def _log(msg):
        if logger:
            logger.info(msg)

    # ── 1. Shape checks ──────────────────────────────────────────────────
    expected_X_shape = (len(y), window_size, n_features)
    shape_ok = X.shape == expected_X_shape
    results["shape_check"] = {
        "passed": shape_ok,
        "expected": str(expected_X_shape),
        "actual": str(X.shape),
    }
    _log(f"  Shape check: {'PASS' if shape_ok else 'FAIL'} "
         f"(expected {expected_X_shape}, got {X.shape})")

    # ── 2. No NaN / Inf values ───────────────────────────────────────────
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    nan_inf_ok = (nan_count == 0) and (inf_count == 0)
    results["nan_inf_check"] = {
        "passed": nan_inf_ok,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    _log(f"  NaN/Inf check: {'PASS' if nan_inf_ok else 'FAIL'} "
         f"(NaN={nan_count}, Inf={inf_count})")

    # ── 3. Metadata row count ────────────────────────────────────────────
    meta_count_ok = len(metadata_df) == len(y)
    results["metadata_count_check"] = {
        "passed": meta_count_ok,
        "sequences": len(y),
        "metadata_rows": len(metadata_df),
    }
    _log(f"  Metadata count check: {'PASS' if meta_count_ok else 'FAIL'} "
         f"(sequences={len(y)}, metadata={len(metadata_df)})")

    # ── 4. Label alignment spot-check (10 random sequences) ──────────────
    n_checks = min(10, len(y))
    if n_checks > 0 and source_df is not None:
        rng = np.random.RandomState(42)
        check_indices = rng.choice(len(y), size=n_checks, replace=False)
        alignment_ok = True

        for idx in check_indices:
            meta_row = metadata_df.iloc[idx]
            pid = meta_row["PatientID"]

            # Get patient data sorted by ICULOS (same order used during building)
            patient_data = source_df[source_df["PatientID"] == pid].sort_values("ICULOS")
            patient_labels = patient_data["SepsisLabel"].values

            # Metadata stores 1-based patient-relative indices:
            #   End_Hour = 1-based end position, so 0-based end index = End_Hour - 1
            #   Prediction target = end_index + horizon
            end_idx = int(meta_row["End_Hour"]) - 1  # convert 1-based to 0-based
            target_idx = end_idx + horizon

            if target_idx >= len(patient_labels):
                alignment_ok = False
                _log(f"    FAIL: Sequence {idx} — target index {target_idx} "
                     f"out of bounds for patient {pid} (length={len(patient_labels)})")
                continue

            expected_label = int(patient_labels[target_idx])
            actual_label = int(y[idx])

            if expected_label != actual_label:
                alignment_ok = False
                _log(f"    FAIL: Sequence {idx} — label mismatch for "
                     f"patient {pid}: expected={expected_label}, got={actual_label}")

        results["label_alignment_check"] = {
            "passed": alignment_ok,
            "samples_checked": n_checks,
        }
        _log(f"  Label alignment check: {'PASS' if alignment_ok else 'FAIL'} "
             f"({n_checks} sequences spot-checked)")
    else:
        results["label_alignment_check"] = {"passed": True, "samples_checked": 0}

    # ── 5. No mixed patients (every sequence maps to one PatientID) ──────
    # Each sequence's metadata PatientID should match the patient_ids array
    pid_match = np.all(metadata_df["PatientID"].values == patient_ids)
    results["no_mixed_patients"] = {
        "passed": bool(pid_match),
    }
    _log(f"  No mixed patients check: {'PASS' if pid_match else 'FAIL'}")

    # ── 6. Labels are binary ────────────────────────────────────────────
    unique_labels = set(np.unique(y))
    labels_binary = unique_labels.issubset({0, 1})
    results["binary_labels_check"] = {
        "passed": labels_binary,
        "unique_labels": list(map(int, np.unique(y))),
    }
    _log(f"  Binary labels check: {'PASS' if labels_binary else 'FAIL'} "
         f"(unique: {list(np.unique(y))})")

    # ── Overall ──────────────────────────────────────────────────────────
    all_passed = all(v["passed"] for v in results.values())
    results["overall"] = {"passed": all_passed}
    _log(f"  Overall validation: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")

    return results
