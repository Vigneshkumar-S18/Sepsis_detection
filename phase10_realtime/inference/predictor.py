# Real-Time PyTorch Inference Predictor Engine
import os
import sys
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.config import MODEL_HYPERPARAMS, DEVICE
from deep_learning.models.bilstm import BiLSTMClassifier
from phase10_realtime.config import MODEL_PATH


class RealTimePredictor:
    """
    Loads model weights once at startup and performs sequential evaluation.
    """
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model weights not found at: {MODEL_PATH}")
            
        logger = None
        self.device = DEVICE
        
        # Instantiate and load model
        self.model = BiLSTMClassifier(**MODEL_HYPERPARAMS["bilstm"])
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

    def predict_risk(self, sequence_95: np.ndarray) -> float:
        """
        Accepts processed sequence (12, 95), runs inference, and returns probability.
        """
        # Format tensor with batch dimension: (1, 12, 95)
        seq_tensor = torch.tensor(sequence_95, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logit = self.model(seq_tensor)
            prob = torch.sigmoid(logit).item()
            
        return prob
