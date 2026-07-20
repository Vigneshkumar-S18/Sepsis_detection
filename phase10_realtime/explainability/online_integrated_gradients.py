# Real-Time Integrated Gradients Attribution Engine
import os
import sys
import torch
import numpy as np
from captum.attr import IntegratedGradients

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deep_learning.config import DEVICE
from explainability.data_loader import get_feature_names


class OnlineIGExplainer:
    """
    Computes real-time feature attributions for a single patient trajectory.
    """
    def __init__(self, model):
        self.model = model
        self.device = DEVICE
        self.ig = IntegratedGradients(model)
        self.feature_names = get_feature_names()

    def attribute_sequence(self, sequence_95: np.ndarray):
        """
        Calculates local attributions and lists top 5 contributing features.
        """
        patient_tensor = torch.tensor(sequence_95, dtype=torch.float32, device=self.device).unsqueeze(0)
        baseline = torch.zeros_like(patient_tensor)
        
        # Run attribution
        attributions = self.ig.attribute(patient_tensor, baseline, target=None)
        attributions = attributions.detach().cpu().numpy()[0]  # shape: (12, 95)
        
        # Sum absolute attribution over time to rank features
        feat_contrib = np.sum(np.abs(attributions), axis=0)
        top_indices = np.argsort(feat_contrib)[-5:][::-1]
        
        top_features = [self.feature_names[idx] for idx in top_indices]
        top_scores = [float(feat_contrib[idx]) for idx in top_indices]
        
        return {
            "top_features": top_features,
            "attribution_scores": top_scores,
            "temporal_attributions": attributions.tolist() # shape: (12, 95)
        }
