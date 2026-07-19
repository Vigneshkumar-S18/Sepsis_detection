# Integrated Gradients Attribution for BiLSTM Sequence Model
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from deep_learning.config import MODEL_HYPERPARAMS
from deep_learning.models.bilstm import BiLSTMClassifier
from explainability.config import VISUALIZATIONS_DIR, OUTPUTS_DIR, DEVICE
from explainability.data_loader import load_sequence_data, get_feature_names


def run_integrated_gradients():
    logger.info("Starting Module 2: Integrated Gradients on BiLSTM Model...")

    # 1. Load model checkpoint
    model_path = os.path.join(project_root, "experiments", "checkpoints", "bilstm_w12_h0_final_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BiLSTM final best model not found at: {model_path}")
        
    model = BiLSTMClassifier(**MODEL_HYPERPARAMS["bilstm"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    # 2. Load sequence data (Test split)
    X, y, pids = load_sequence_data("w12_h0", "test")
    feature_names = get_feature_names()

    # Find positive instances (sepsis patients) to calculate positive class attributions
    pos_indices = np.where(y == 1)[0]
    if len(pos_indices) == 0:
        # Fallback to general indices if no positives found
        pos_indices = np.arange(len(y))

    # Sample 100 positive sequences to compute average attributions
    np.random.seed(SEED := 42)
    sample_indices = np.random.choice(pos_indices, size=min(100, len(pos_indices)), replace=False)
    X_sample = torch.tensor(X[sample_indices], dtype=torch.float32, device=DEVICE)
    
    # 3. Compute Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Define baseline as zeros (no temporal features measurement)
    baseline = torch.zeros_like(X_sample)
    
    # Run attribution
    attributions, delta = ig.attribute(
        X_sample, baseline, target=None, return_convergence_delta=True
    )
    
    attributions = attributions.detach().cpu().numpy()  # shape: (N, seq_len, features)

    # 4. Compute Time × Feature Attribution Matrix
    # Take the mean of absolute attribution values across samples
    mean_abs_attrib = np.mean(np.abs(attributions), axis=0)  # shape: (seq_len, features)
    
    # Global feature rankings: sum attributions over the time-steps and take average
    global_attrib = np.mean(np.sum(np.abs(attributions), axis=1), axis=0)
    ig_ranking_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_Attribution": global_attrib
    }).sort_values(by="Mean_Abs_Attribution", ascending=False).reset_index(drop=True)

    # Save rankings
    ranking_path = os.path.join(OUTPUTS_DIR, "ig_feature_rankings.csv")
    ig_ranking_df.to_csv(ranking_path, index=False)
    logger.info(f"  Saved Integrated Gradients rankings to: {ranking_path}")

    # 5. Plot Time × Feature Heatmap for top 15 features
    top_15_features = ig_ranking_df["Feature"].head(15).tolist()
    top_15_indices = [feature_names.index(f) for f in top_15_features]
    
    # Slice the temporal attribution matrix for top 15 features
    heatmap_data = mean_abs_attrib[:, top_15_indices].T  # shape: (15, seq_len)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".4f", cmap="Oranges",
        xticklabels=[f"t-{12-t}h" for t in range(12)],
        yticklabels=top_15_features
    )
    plt.title("BiLSTM Temporal Attribution Heatmap (Integrated Gradients)", fontsize=12, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("Observation Timeline Timeline Step")
    plt.ylabel("Top 15 Influential Features")
    plt.tight_layout()
    
    heatmap_path = os.path.join(VISUALIZATIONS_DIR, "ig_temporal_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    logger.info(f"  Saved Temporal Attribution Heatmap to: {heatmap_path}")

    return ig_ranking_df, attributions, sample_indices


if __name__ == "__main__":
    run_integrated_gradients()
