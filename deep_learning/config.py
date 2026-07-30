# Central Configuration for Deep Temporal Learning Models
import torch

SEED = 42

# Sequence details
FEATURES = 391  # Total features input to sequential neural networks


# Optimizer, Loss, and Scheduler
OPTIMIZER = "Adam"
LOSS_FUNCTION = "BCEWithLogits"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# High Sepsis class imbalance scaling factor (approx 98.2% non-sepsis to 1.8% sepsis)
POS_WEIGHT = 54.5

# Early stopping & scheduler parameters
EARLY_STOPPING_PATIENCE = 3  # Stop if validation AUPRC does not improve for 3 epochs
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 1

# Hardware setup - Default to CPU since GPU/CUDA is not available
DEVICE = "cpu"

# Training speed optimization configurations for CPU execution
BATCH_SIZE = 1024
EPOCHS = 8  # Keep epochs minimal for speed during evaluation

# Model architectures hyperparameter mappings
MODEL_HYPERPARAMS = {
    "lstm": {
        "input_dim": FEATURES,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2
    },
    "gru": {
        "input_dim": FEATURES,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2
    },
    "bilstm": {
        "input_dim": FEATURES,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2
    },
    "transformer": {
        "input_dim": FEATURES,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1
    }
}
