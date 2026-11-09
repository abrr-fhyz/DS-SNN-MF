"""MLP encoder for the clinical / demographic table."""
import torch.nn as nn
from torch import Tensor

from config import cfg


class ClinicalMLP(nn.Module):
    """MLP encoder for the clinical/demographic features."""
    def __init__(self, n_features: int, out_dim: int = cfg.CLINICAL_FEAT_DIM, dropout: float = cfg.ENCODER_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
