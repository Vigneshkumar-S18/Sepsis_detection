# BiLSTM Sequence Model Hyperparameter Optimization Module
import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from deep_learning.config import POS_WEIGHT, DEVICE
from deep_learning.models.bilstm import BiLSTMClassifier
from deep_learning.trainer import SepsisDeepTrainer
from deep_learning.run_training import set_all_seeds
from optimization.config import BILSTM_PARAM_OPTIONS, OUTPUTS_DIR, SEED
from optimization.data_loader import load_sequence_splits


def run_bilstm_optimization(num_trials=5):
    logger.info("Running hyperparameter optimization for BiLSTM sequence model...")
    
    # Load raw numpy sequences
    train_split, val_split, _ = load_sequence_splits("w12_h0")
    X_train, y_train, _ = train_split
    X_val, y_val, _ = val_split

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    # Downsample sequences to 20% representative split to run optimization fast on CPU
    np.random.seed(SEED)
    train_size = int(len(X_train) * 0.20)
    val_size = int(len(X_val) * 0.20)
    
    train_idx = np.random.choice(len(X_train), size=train_size, replace=False)
    val_idx = np.random.choice(len(X_val), size=val_size, replace=False)
    
    X_train_sub, y_train_sub = X_train[train_idx], y_train[train_idx]
    X_val_sub, y_val_sub = X_val[val_idx], y_val[val_idx]

    # Generate trials: Force default baseline model configuration as Trial 1
    trials = []
    trials.append({
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 1024
    })

    for trial_idx in range(1, num_trials):
        trial = {}
        for param, options in BILSTM_PARAM_OPTIONS.items():
            trial[param] = np.random.choice(options)
        # Ensure correct type conversion
        trial["hidden_size"] = int(trial["hidden_size"])
        trial["num_layers"] = int(trial["num_layers"])
        trial["dropout"] = float(trial["dropout"])
        trial["learning_rate"] = float(trial["learning_rate"])
        trial["batch_size"] = int(trial["batch_size"])
        trials.append(trial)

    results = []
    best_auprc = -1.0
    best_params = None

    for idx, params in enumerate(trials):
        logger.info(f"  Trial {idx+1}/{num_trials} — Parameters: {params}")
        
        # Prepare datasets
        train_ds = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32), torch.tensor(y_train_sub, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val_sub, dtype=torch.float32), torch.tensor(y_val_sub, dtype=torch.float32))
        
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

        set_all_seeds(SEED)
        start_time = time.time()
        
        # Initialize model
        model = BiLSTMClassifier(
            input_dim=95,
            hidden_dim=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        
        trainer = SepsisDeepTrainer(
            model=model, device=DEVICE, criterion=criterion, optimizer=optimizer,
            scheduler=None, early_stopping_patience=2, checkpoints_dir=None, logger=logger
        )
        
        # Train for 2 short epochs
        trainer.fit(train_loader, val_loader, max_epochs=2, checkpoint_name=f"opt_trial_{idx+1}")
        
        # Evaluate on validation split
        val_loss, val_auroc, val_auprc, _, _ = trainer.evaluate(val_loader)
        
        elapsed = time.time() - start_time
        logger.info(f"    Validation AUPRC: {val_auprc:.4f} | Validation AUROC: {val_auroc:.4f} | Duration: {elapsed:.1f}s")
        
        trial_result = {
            "trial": idx + 1,
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "dropout": params["dropout"],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "duration_sec": elapsed
        }
        results.append(trial_result)
        
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_params = params
            
        # Clean memory
        del model, trainer
        gc.collect()

    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUTS_DIR, "bilstm_trials.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"  BiLSTM trials logged to: {results_path}")
    logger.info(f"  Best BiLSTM Parameters found: {best_params} (Val AUPRC: {best_auprc:.4f})")
    
    return best_params, best_auprc


if __name__ == "__main__":
    run_bilstm_optimization()
