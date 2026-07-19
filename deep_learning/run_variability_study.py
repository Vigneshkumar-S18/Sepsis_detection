# variability study running BiLSTM on w12_h0 across multiple seeds
import os
import sys
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from deep_learning.config import (
    BATCH_SIZE, LEARNING_RATE, POS_WEIGHT,
    DEVICE, MODEL_HYPERPARAMS, SCHEDULER_FACTOR, SCHEDULER_PATIENCE
)
from deep_learning.data_loader import get_sequence_dataloaders
from deep_learning.models.bilstm import BiLSTMClassifier
from deep_learning.trainer import SepsisDeepTrainer
from deep_learning.run_training import set_all_seeds


def run_variability_study():
    logger.info("Running deep learning statistical variability study (3 seeds)...")
    
    cfg_seq_dir = os.path.join(project_root, "datasets", "sequences", "w12_h0")
    checkpoints_dir = os.path.join(project_root, "experiments", "checkpoints")
    
    seeds = [42, 100, 2026]
    results = []

    # Seed 42 is already run, but to ensure consistency of training splits
    # we will re-run it or run all 3 seeds sequentially. Since each run is only
    # ~2.5 mins, running all 3 sequentially takes less than 8 minutes and guarantees
    # exact reproducibility on the 50% data sample.
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    max_epochs = 4

    # Load 50% sample dataloaders
    train_loader, val_loader, test_loader = get_sequence_dataloaders(
        cfg_seq_dir, BATCH_SIZE, sample_pct=0.50, logger=logger
    )

    for seed in seeds:
        logger.info(f"\n--- Training BiLSTM on w12_h0 with Seed: {seed} ---")
        set_all_seeds(seed)

        model = BiLSTMClassifier(**MODEL_HYPERPARAMS["bilstm"])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )

        trainer = SepsisDeepTrainer(
            model=model, device=DEVICE, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, early_stopping_patience=2,
            checkpoints_dir=checkpoints_dir, logger=logger
        )

        trainer.fit(train_loader, val_loader, max_epochs, f"bilstm_w12_h0_seed_{seed}")

        # Evaluate test metrics
        test_loss, test_auroc, test_auprc, _, _ = trainer.evaluate(test_loader)
        logger.info(f"  Seed {seed} Test Results — AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}")

        results.append({
            "seed": seed,
            "auroc": test_auroc,
            "auprc": test_auprc
        })

        # Cleanup memory
        del model, trainer
        gc.collect()

    # Calculate statistics
    aurocs = [r["auroc"] for r in results]
    auprcs = [r["auprc"] for r in results]

    mean_auroc = float(np.mean(aurocs))
    std_auroc = float(np.std(aurocs))
    mean_auprc = float(np.mean(auprcs))
    std_auprc = float(np.std(auprcs))

    stats = {
        "runs": results,
        "auroc_mean": round(mean_auroc, 4),
        "auroc_std": round(std_auroc, 4),
        "auprc_mean": round(mean_auprc, 4),
        "auprc_std": round(std_auprc, 4),
        "formatted_auroc": f"{mean_auroc:.4f} \u00B1 {std_auroc:.4f}",
        "formatted_auprc": f"{mean_auprc:.4f} \u00B1 {std_auprc:.4f}"
    }

    # Save to disk
    save_path = os.path.join(project_root, "experiments", "metrics", "dl_variability_study.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

    logger.info("\n" + "="*70)
    logger.info(f"Variability Study Complete! saved to {save_path}")
    logger.info(f"BiLSTM AUROC: {stats['formatted_auroc']}")
    logger.info(f"BiLSTM AUPRC: {stats['formatted_auprc']}")
    logger.info("="*70)


if __name__ == "__main__":
    run_variability_study()
