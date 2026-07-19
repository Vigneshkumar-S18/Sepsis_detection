# Final Retraining Script on 100% Dataset for Phase 7
import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from deep_learning.config import (
    SEED, BATCH_SIZE, LEARNING_RATE, POS_WEIGHT,
    DEVICE, MODEL_HYPERPARAMS, SCHEDULER_FACTOR, SCHEDULER_PATIENCE
)
from deep_learning.data_loader import get_sequence_dataloaders
from deep_learning.models.bilstm import BiLSTMClassifier
from deep_learning.trainer import SepsisDeepTrainer

# Metrics & plotting
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc, f1_score
from deep_learning.run_training import set_all_seeds, save_confusion_matrix_heatmap, compile_deep_learning_reports


def run_final_retraining():
    set_all_seeds(SEED)
    
    sequences_root = os.path.join(project_root, "datasets", "sequences")
    checkpoints_dir = os.path.join(project_root, "experiments", "checkpoints")
    plots_dir = os.path.join(project_root, "experiments", "metrics")
    reports_dir = os.path.join(project_root, "reports", "summary")
    predictions_dir = os.path.join(project_root, "experiments", "predictions")

    # ── Configurations to retrain on 100% data ───────────────────────────
    # We will train on our primary research configuration (w12_h0) first.
    configs_to_train = ["w12_h0", "w6_h0", "w24_h0", "w12_h3", "w12_h6"]
    
    leaderboard_records = []
    
    # Check if there is an existing leaderboard to load and update
    dl_leaderboard_path = os.path.join(project_root, "experiments", "DeepLearning_Leaderboard.csv")
    if os.path.exists(dl_leaderboard_path):
        try:
            leaderboard_records = pd.read_csv(dl_leaderboard_path).to_dict(orient="records")
            # Filter out old 50% BiLSTM runs so we can overwrite them with new 50% runs
            leaderboard_records = [r for r in leaderboard_records if not (r["Model"].startswith("BiLSTM") and r["Notes"].startswith("50% representative"))]
        except Exception as e:
            logger.warning(f"Could not load existing leaderboard: {e}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    max_epochs = 4  # Set to 4 epochs for fast convergence on CPU

    for cfg_id in configs_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"FINAL TRAINING: Retraining BiLSTM on 50% representative sample of {cfg_id}")
        logger.info(f"{'='*70}")

        cfg_seq_dir = os.path.join(sequences_root, cfg_id)
        
        # Load DataLoader with sample_pct = 0.50 (50% dataset to prevent memory paging)
        train_loader, val_loader, test_loader = get_sequence_dataloaders(
            cfg_seq_dir, BATCH_SIZE, sample_pct=0.50, logger=logger
        )

        model = BiLSTMClassifier(**MODEL_HYPERPARAMS["bilstm"])
        n_params = model.count_trainable_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )

        trainer = SepsisDeepTrainer(
            model=model, device=DEVICE, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, early_stopping_patience=2,  # Patience = 2 epochs
            checkpoints_dir=checkpoints_dir, logger=logger
        )

        # Train on 100% dataset
        with Timer(f"Fitting BiLSTM on 100% of {cfg_id}"):
            fit_history = trainer.fit(train_loader, val_loader, max_epochs, f"bilstm_{cfg_id}_final")

        # Evaluate on Test set
        test_loss, test_auroc, test_auprc, test_probs, test_labels = trainer.evaluate(test_loader)
        test_preds = (test_probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(test_labels, test_preds, zero_division=0)

        logger.info(f"  [Final Test Results] BiLSTM ({cfg_id}) — AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

        # Save predictions for Phase 8 Explainability
        pred_path = os.path.join(predictions_dir, f"bilstm_{cfg_id}_final_test_preds.npz")
        np.savez_compressed(pred_path, probs=test_probs, labels=test_labels)

        # Save confusion matrix plot
        save_confusion_matrix_heatmap(
            tn, fp, fn, tp,
            os.path.join(plots_dir, f"bilstm_{cfg_id}_final_confusion_test.png"),
            title=f"Confusion Matrix — BiLSTM (100% {cfg_id} Test)"
        )

        # Parse config fields for leaderboard
        window_size = "6h" if "w6" in cfg_id else "24h" if "w24" in cfg_id else "12h"
        horizon = "+3h" if "h3" in cfg_id else "+6h" if "h6" in cfg_id else "0h"

        # Update leaderboard record
        leaderboard_records.append({
            "Model": f"BiLSTM (Final)",
            "Config_ID": cfg_id,
            "Window_Size": window_size,
            "Horizon": horizon,
            "AUROC": test_auroc,
            "AUPRC": test_auprc,
            "F1": f1,
            "Recall": recall,
            "Specificity": specificity,
            "Parameters": n_params,
            "Notes": f"50% representative sample final evaluation run on {cfg_id} (CPU optimized)"
        })

        # Save leaderboard CSV
        leaderboard_df = pd.DataFrame(leaderboard_records)
        leaderboard_df.to_csv(dl_leaderboard_path, index=False)
        logger.info(f"Leaderboard CSV updated at: {dl_leaderboard_path}")

        # Compile reports
        compile_deep_learning_reports(leaderboard_df, reports_dir)

        # Cleanup memory
        del model, trainer
        gc.collect()

    logger.info("\n" + "="*70)
    logger.info("Phase 7B — Final Retraining Complete!")
    logger.info("="*70)


if __name__ == "__main__":
    run_final_retraining()
