# Parse logs and plot learning curves for Phase 7
import os
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger


def parse_and_plot_curves():
    log_path = os.path.join(
        "C:\\Users\\VIGNESH KUMAR\\.gemini\\antigravity-ide\\brain\\49add639-4f4e-4273-83fb-5994bf7082c1\\.system_generated\\tasks\\task-1278.log"
    )
    plots_dir = os.path.join(project_root, "experiments", "metrics")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(log_path):
        logger.error(f"Development task log not found at: {log_path}")
        return

    # Dictionary to hold history for each model
    histories = {}
    current_model = None

    # Read development log line by line
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Detect model training start
            model_start_match = re.search(r"Milestone 7A: Training (\w+) on w12_h0", line)
            if model_start_match:
                current_model = model_start_match.group(1)
                histories[current_model] = {
                    "train_loss": [], "val_loss": [], "val_auprc": [], "val_auroc": []
                }
                continue

            # Detect epoch metrics line
            epoch_match = re.search(
                r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Val AUPRC: ([\d.]+) \| Val AUROC: ([\d.]+)",
                line
            )
            if epoch_match and current_model:
                histories[current_model]["train_loss"].append(float(epoch_match.group(2)))
                histories[current_model]["val_loss"].append(float(epoch_match.group(3)))
                histories[current_model]["val_auprc"].append(float(epoch_match.group(4)))
                histories[current_model]["val_auroc"].append(float(epoch_match.group(5)))

    # Plot learning curves for each model
    sns.set_theme(style="whitegrid")
    
    for name, history in histories.items():
        if not history["train_loss"]:
            continue
            
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(11, 4.5))
        
        # Loss curves subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o', color='#3b82f6', lw=1.5)
        plt.plot(epochs, history["val_loss"], label="Val Loss", marker='s', color='#ef4444', lw=1.5)
        plt.title(f"{name} — Loss Curves", fontsize=11, weight='bold', color='#0f766e')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(epochs)
        plt.legend(fontsize=9)
        
        # Metrics curves subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["val_auprc"], label="Val AUPRC", marker='^', color='#f59e0b', lw=1.5)
        plt.plot(epochs, history["val_auroc"], label="Val AUROC", marker='d', color='#10b981', lw=1.5)
        plt.title(f"{name} — Validation Metrics", fontsize=11, weight='bold', color='#0f766e')
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim([0.0, 1.05])
        plt.xticks(epochs)
        plt.legend(fontsize=9, loc="lower left")
        
        plt.tight_layout()
        save_path = os.path.join(plots_dir, f"{name.lower()}_learning_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Learning curves saved for {name} at: {save_path}")


if __name__ == "__main__":
    parse_and_plot_curves()
