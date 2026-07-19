# Measure Model Complexity and Inference Efficiency
import os
import sys
import time
import torch
import numpy as np
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.config import MODEL_HYPERPARAMS
from deep_learning.models.lstm import LSTMClassifier
from deep_learning.models.gru import GRUClassifier
from deep_learning.models.bilstm import BiLSTMClassifier
from deep_learning.models.transformer import TransformerClassifier


def measure_efficiency():
    print("Measuring deep learning model complexity and efficiency...")
    
    models = {
        "LSTM": LSTMClassifier(**MODEL_HYPERPARAMS["lstm"]),
        "GRU": GRUClassifier(**MODEL_HYPERPARAMS["gru"]),
        "BiLSTM": BiLSTMClassifier(**MODEL_HYPERPARAMS["bilstm"]),
        "Transformer": TransformerClassifier(**MODEL_HYPERPARAMS["transformer"])
    }
    
    results = {}
    batch_size = 1024
    seq_len = 12
    features = 95
    device = "cpu"
    
    # Generate dummy input batch
    dummy_input = torch.randn(batch_size, seq_len, features, device=device)
    
    for name, model in models.items():
        model.eval()
        
        # 1. Parameter count
        n_params = model.count_trainable_parameters()
        
        # 2. Model size on disk (approximate via temporary state dict save)
        temp_path = f"temp_{name.lower()}_model.pt"
        torch.save(model.state_dict(), temp_path)
        model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # 3. Inference time per batch and per sample (warm up first, then measure average of 50 runs)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
                
            t0 = time.perf_counter()
            runs = 50
            for _ in range(runs):
                _ = model(dummy_input)
            total_time = time.perf_counter() - t0
            
            avg_batch_time_ms = (total_time / runs) * 1000
            avg_sample_time_us = (avg_batch_time_ms / batch_size) * 1000
            
        print(f"  {name:12} | Params: {n_params:7,} | Size: {model_size_mb:.4f} MB | Batch: {avg_batch_time_ms:6.2f} ms | Sample: {avg_sample_time_us:5.2f} us")
        
        results[name] = {
            "trainable_parameters": n_params,
            "model_size_mb": round(model_size_mb, 4),
            "inference_time_batch_ms": round(avg_batch_time_ms, 2),
            "inference_time_sample_us": round(avg_sample_time_us, 2)
        }
        
    # Save metrics JSON
    metrics_path = os.path.join(project_root, "experiments", "metrics", "dl_complexity_efficiency.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nComplexity & efficiency metrics saved to: {metrics_path}")


if __name__ == "__main__":
    measure_efficiency()
