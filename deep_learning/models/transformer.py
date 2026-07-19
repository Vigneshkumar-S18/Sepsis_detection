# Transformer Encoder Sequence Classifier Model
import torch
import torch.nn as nn
import math
from deep_learning.models.base_model import BaseSepsisModel


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding injection to add sequential time-order information
    to the permutation-invariant Transformer attention weights.
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices; cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd dimension edge cases safely
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(BaseSepsisModel):
    """
    Transformer Encoder Classifier.
    Projects raw features, injects positional encoding, passes through
    self-attention encoder layers, and pools temporally.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        # Project input dimension (95) to Transformer embedding dimension (d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x_proj = self.input_projection(x)  # shape: (batch, seq_len, d_model)
        x_pos = self.pos_encoder(x_proj)
        
        # Attention forward pass
        out = self.transformer_encoder(x_pos)  # shape: (batch, seq_len, d_model)
        
        # Temporal Average Pooling to consolidate sequence timeline representation
        pooled = out.mean(dim=1)  # shape: (batch, d_model)
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  # shape: (batch, 1)
        
        return logits.squeeze(-1)  # shape: (batch,)
