# Configuration settings for Phase 9 Hyperparameter Optimization
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directories
LOGS_DIR = os.path.join(project_root, "optimization", "logs")
METRICS_DIR = os.path.join(project_root, "optimization", "metrics")
OUTPUTS_DIR = os.path.join(project_root, "optimization", "outputs")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

SEED = 42

# XGBoost search space definitions
XGB_PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# BiLSTM search space definitions
BILSTM_PARAM_OPTIONS = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 5e-4, 2e-3],
    "batch_size": [512, 1024]
}
