# Bidirectional LSTM Sequence Classifier Model
import torch
import torch.nn as nn
from deep_learning.models.base_model import BaseSepsisModel


class BiLSTMClassifier(BaseSepsisModel):
    """
    Bidirectional LSTM Classifier.
    Captures temporal progression in both forward and reverse directions.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(BiLSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer input dimension is doubled due to bidirectionality (hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last temporal step containing both directions
        # out shape: (batch, seq_len, hidden_dim * 2)
        last_step = out[:, -1, :]  # shape: (batch, hidden_dim * 2)
        
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)  # shape: (batch, 1)
        
        return logits.squeeze(-1)  # shape: (batch,)
