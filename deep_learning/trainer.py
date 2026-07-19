# Reusable Neural Network Trainer with validation AUPRC early stopping
import os
import time
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


class SepsisDeepTrainer:
    """
    Standard PyTorch training coordinator. Trains sequence classifiers,
    evaluates validation epochs, tracks early-stopping patience on AUPRC,
    and checkpoint saving.
    """
    def __init__(self, model, device, criterion, optimizer, scheduler=None,
                 early_stopping_patience=3, checkpoints_dir=None, logger=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = early_stopping_patience
        self.checkpoints_dir = checkpoints_dir
        self.logger = logger

        # Early stopping tracking
        self.best_val_auprc = -1.0
        self.best_model_state = None
        self.patience_counter = 0

        # Loss and metric histories
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auprc": [],
            "val_auroc": []
        }

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        n_batches = len(dataloader)

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_batches

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                # Sigmoid activation to transform logits to probabilities
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Calculate ROC and AUPRC
        auroc = float(roc_auc_score(all_labels, all_probs))
        precisions, recalls, _ = precision_recall_curve(all_labels, all_probs)
        auprc = float(auc(recalls, precisions))

        return avg_loss, auroc, auprc, all_probs, all_labels

    def fit(self, train_loader, val_loader, max_epochs, checkpoint_name):
        self.logger.info(f"Starting model fit: {checkpoint_name} (Max Epochs: {max_epochs}, Device: {self.device})")
        
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auroc, val_auprc, _, _ = self.evaluate(val_loader)
            epoch_time = time.time() - t0

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auprc"].append(val_auprc)
            self.history["val_auroc"].append(val_auroc)

            self.logger.info(
                f"  Epoch {epoch:02d}/{max_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUPRC: {val_auprc:.4f} | Val AUROC: {val_auroc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Check LR Scheduler
            if self.scheduler:
                # Most schedulers adjust on validation loss or validation metrics
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Monitor validation AUPRC (minimize negative AUPRC or maximize AUPRC)
                    self.scheduler.step(val_loss)

            # Early stopping check on validation AUPRC (maximize)
            if val_auprc > self.best_val_auprc:
                self.best_val_auprc = val_auprc
                # Deep copy model state dict
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0

                # Save temporary best checkpoint
                if self.checkpoints_dir:
                    os.makedirs(self.checkpoints_dir, exist_ok=True)
                    best_path = os.path.join(self.checkpoints_dir, f"{checkpoint_name}_best.pt")
                    torch.save(self.best_model_state, best_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f"  Early stopping triggered! Restoring best model from epoch {epoch - self.patience}")
                    break

        # Restore best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
