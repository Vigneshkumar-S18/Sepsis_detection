# ─────────────────────────────────────────────────────────────────────────────
# Metadata — Builds per-sequence metadata records for traceability
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd


def build_sequence_metadata_record(patient_id, start_hour, end_hour,
                                   prediction_hour, window_size, horizon,
                                   label, split_name):
    """
    Creates a single metadata record dict for one generated sequence.

    Parameters
    ----------
    patient_id : str
        The patient identifier.
    start_hour : int
        ICULOS value at the start of the window.
    end_hour : int
        ICULOS value at the end of the window.
    prediction_hour : int
        ICULOS value at which the label is taken (end_hour + horizon).
    window_size : int
        Length of the observation window in hours.
    horizon : int
        Prediction horizon in hours.
    label : int
        Binary label (0 or 1).
    split_name : str
        Dataset split name (train, validation, test).

    Returns
    -------
    dict
        A metadata record.
    """
    return {
        "PatientID": patient_id,
        "Start_Hour": int(start_hour),
        "End_Hour": int(end_hour),
        "Prediction_Hour": int(prediction_hour),
        "Window_Size": int(window_size),
        "Prediction_Horizon": int(horizon),
        "Label": int(label),
        "Split": split_name,
    }


def metadata_list_to_dataframe(metadata_list):
    """
    Converts a list of metadata dicts into a Pandas DataFrame.
    """
    return pd.DataFrame(metadata_list)
