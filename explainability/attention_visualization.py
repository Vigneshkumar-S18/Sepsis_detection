# Attention Weight Visualization for the Transformer Model
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from deep_learning.config import MODEL_HYPERPARAMS
from deep_learning.models.transformer import TransformerClassifier
from explainability.config import VISUALIZATIONS_DIR, DEVICE
from explainability.data_loader import load_sequence_data


def run_attention_visualization():
    logger.info("Starting Module 3: Transformer Attention Visualization...")

    # 1. Load model checkpoint
    model_path = os.path.join(project_root, "experiments", "checkpoints", "transformer_w12_h0_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer model checkpoint not found at: {model_path}")
        
    model = TransformerClassifier(**MODEL_HYPERPARAMS["transformer"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    # 2. Load sequence data (Test split)
    X, y, pids = load_sequence_data("w12_h0", "test")

    # Find positive instances (sepsis patients)
    pos_indices = np.where(y == 1)[0]
    if len(pos_indices) == 0:
        pos_indices = np.arange(len(y))

    # Take a representative positive patient sequence
    np.random.seed(42)
    sample_idx = np.random.choice(pos_indices)
    x_patient = torch.tensor(X[sample_idx], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # shape: (1, 12, 95)

    # 3. Extract Self-Attention weights mathematically
    # Since PyTorch's default TransformerEncoder forward pass hides attention scores,
    # we replicate the self-attention dot-product mechanism using the trained weights.
    with torch.no_grad():
        # Project raw patient trajectory into Transformer embedding space (d_model = 64)
        x_proj = model.input_projection(x_patient)  # shape: (1, 12, 64)
        x_pos = model.pos_encoder(x_proj)
        
        # Access first self-attention layer
        attn_layer = model.transformer_encoder.layers[0].self_attn
        d_model = attn_layer.embed_dim
        nhead = attn_layer.num_heads
        head_dim = d_model // nhead  # 64 / 4 = 16
        
        # Extract in_proj_weight (contains weights for Query, Key, and Value projections concatenated)
        in_proj_weight = attn_layer.in_proj_weight  # shape: (3*d_model, d_model) = (192, 64)
        q_weight, k_weight, v_weight = torch.chunk(in_proj_weight, 3, dim=0)  # shape of each: (64, 64)
        
        in_proj_bias = attn_layer.in_proj_bias
        q_bias, k_bias, v_bias = torch.chunk(in_proj_bias, 3, dim=0)

        # Compute Q and K projections
        Q = torch.matmul(x_pos, q_weight.t()) + q_bias  # shape: (1, 12, 64)
        K = torch.matmul(x_pos, k_weight.t()) + k_bias  # shape: (1, 12, 64)

        # Reshape for multi-head attention (batch_size, nhead, seq_len, head_dim)
        Q = Q.view(1, 12, nhead, head_dim).transpose(1, 2)  # shape: (1, 4, 12, 16)
        K = K.view(1, 12, nhead, head_dim).transpose(1, 2)  # shape: (1, 4, 12, 16)

        # Compute raw dot-product score weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)  # shape: (1, 4, 12, 12)
        attn_weights = torch.softmax(attn_scores, dim=-1).cpu().numpy()[0]  # shape: (4, 12, 12)

    # 4. Save Attention Heatmap (Average across all 4 attention heads)
    avg_attention = np.mean(attn_weights, axis=0)  # shape: (12, 12)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        avg_attention, annot=True, fmt=".2f", cmap="Purples",
        xticklabels=[f"t-{12-t}h" for t in range(12)],
        yticklabels=[f"t-{12-t}h" for t in range(12)]
    )
    plt.title("Transformer Average Self-Attention Heatmap", fontsize=12, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("Key Step (Attended Hours)")
    plt.ylabel("Query Step (Current Observation Hours)")
    plt.tight_layout()
    
    heatmap_path = os.path.join(VISUALIZATIONS_DIR, "transformer_attention_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    logger.info(f"  Saved Transformer Attention Heatmap to: {heatmap_path}")

    # 5. Plot Temporal Attention distribution (where does it focus overall?)
    # Sum/average attention weight given to each timestep by all query hours
    temporal_focus = np.mean(avg_attention, axis=0)
    
    plt.figure(figsize=(7, 4))
    plt.bar([f"t-{12-t}h" for t in range(12)], temporal_focus, color="#8b5cf6", edgecolor="purple")
    plt.title("Transformer Average Temporal Attention Distribution", fontsize=11, weight='bold', color='#0f766e', pad=15)
    plt.xlabel("Timeline Step")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    
    dist_path = os.path.join(VISUALIZATIONS_DIR, "transformer_temporal_attention.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    logger.info(f"  Saved Temporal Attention Distribution to: {dist_path}")

    return attn_weights, avg_attention


if __name__ == "__main__":
    run_attention_visualization()
