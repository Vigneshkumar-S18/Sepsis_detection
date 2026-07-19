# Configuration for Baseline Classical Machine Learning Models
SEED = 42

EXCLUDE_COLUMNS = ["PatientID", "SepsisLabel"]
LABEL_COLUMN = "SepsisLabel"

# Sepsis class imbalance ratio is ~54.5 (98.2% non-sepsis to 1.8% sepsis)
CLASS_WEIGHTS = {0: 1.0, 1: 54.5}

MODEL_CONFIGS = {
    "logistic_regression": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 100,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1
    },
    "decision_tree": {
        "max_depth": 8,
        "class_weight": "balanced",
        "random_state": SEED
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 8,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "scale_pos_weight": 54.5,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": SEED
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "scale_pos_weight": 54.5,
        "learning_rate": 0.1,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1
    }
}
