# LSTM Sequence Classifier Model
import torch
import torch.nn as nn
from deep_learning.models.base_model import BaseSepsisModel


class LSTMClassifier(BaseSepsisModel):
    """
    Standard sequential LSTM classifier.
    Processes 3D input tensor of shape (batch, sequence_length, features).
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Dense output head producing logits
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last temporal step
        # out shape: (batch, seq_len, hidden_dim)
        last_step = out[:, -1, :]  # shape: (batch, hidden_dim)
        
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)  # shape: (batch, 1)
        
        return logits.squeeze(-1)  # shape: (batch,)
