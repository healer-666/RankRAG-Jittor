"""Jittor TextCNN reranker for query-passage token ids."""

from __future__ import annotations

import os

os.environ.setdefault("use_cuda", "0")
os.environ.setdefault("nvcc_path", "")

try:
    import jittor as jt
    from jittor import nn
except Exception as exc:  # pragma: no cover - depends on local Jittor install.
    jt = None
    nn = None
    JITTOR_IMPORT_ERROR = exc
else:
    if jt is not None:
        jt.flags.use_cuda = 0
    JITTOR_IMPORT_ERROR = None


def require_jittor() -> None:
    if jt is None or nn is None:
        raise RuntimeError(
            "Jittor import failed. Please install Jittor and ensure a compatible CPU build environment. "
            f"Original error: {JITTOR_IMPORT_ERROR}"
        )


if nn is not None:

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
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes])
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)
            self.padding_idx = padding_idx

        def execute(self, input_ids):
            x = self.embedding(input_ids.int64()).transpose(0, 2, 1)
            pooled = []
            for conv in self.convs:
                h = self.relu(conv(x))
                pooled.append(h.max(dim=2))
            x = jt.concat(pooled, dim=1)
            x = self.dropout(x)
            return self.fc(x).squeeze(-1)

else:

    class TextCNNScorer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_jittor()


def pairwise_ranking_loss(score_pos, score_neg):
    require_jittor()
    return -jt.log(jt.sigmoid(score_pos - score_neg) + 1e-8).mean()
