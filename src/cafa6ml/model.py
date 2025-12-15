from __future__ import annotations
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, dropout: float, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        return self.net(x)
