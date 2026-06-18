"""PyTorch MLP scorer for query-candidate ranking features."""

from __future__ import annotations

import torch
from torch import nn


class MLPScorer(nn.Module):
    """A small scorer that maps 1537-dim features to one relevance score."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def pairwise_ranking_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8).mean()
