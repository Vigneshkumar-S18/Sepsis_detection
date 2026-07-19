# ─────────────────────────────────────────────────────────────────────────────
# Sequence Generation Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Sliding window sizes (in hours) to generate sequences for
WINDOW_SIZES = [6, 12, 24]

# Prediction horizons (in hours ahead of window end)
#   0 = current-hour detection
#   3 = 3-hour early warning
#   6 = 6-hour early warning
PREDICTION_HORIZONS = [0, 3, 6]

# Stride for the sliding window (1 = maximum overlap / maximum samples)
STRIDE = 1

# Minimum patient stay length required to generate any sequence
MIN_SEQUENCE_LENGTH = 6

# Label mode: "future" means label is taken at window_end + horizon
LABEL_MODE = "future"

# Columns to EXCLUDE from the sequence feature tensor
EXCLUDE_COLUMNS = ["PatientID", "SepsisLabel"]

# Label column name
LABEL_COLUMN = "SepsisLabel"

# Patient ID column name
PATIENT_ID_COLUMN = "PatientID"

# ─────────────────────────────────────────────────────────────────────────────
# Experiment Configurations
# Each entry maps a dataset ID to its (window_size, prediction_horizon) pair.
# These 5 configurations answer distinct research questions.
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_CONFIGS = {
    "w6_h0":  {"window_size": 6,  "horizon": 0, "description": "Minimal observation: can 6h detect sepsis?"},
    "w12_h0": {"window_size": 12, "horizon": 0, "description": "Standard 12h observation window"},
    "w24_h0": {"window_size": 24, "horizon": 0, "description": "Extended observation: does more history help?"},
    "w12_h3": {"window_size": 12, "horizon": 3, "description": "Early warning: 3-hour advance prediction"},
    "w12_h6": {"window_size": 12, "horizon": 6, "description": "Early warning: 6-hour advance prediction"},
}
