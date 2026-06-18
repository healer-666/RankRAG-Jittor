"""PyTorch TextCNN reranker for query-passage token ids."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TextCNNScorer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids.long()).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            pooled.append(F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2))
        x = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(x).squeeze(-1)


def pairwise_ranking_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8).mean()
