"""Jittor MLP scorer for query-candidate ranking features."""

from __future__ import annotations


try:
    import jittor as jt
    from jittor import nn
except Exception as exc:  # pragma: no cover - depends on local Jittor install.
    jt = None
    nn = None
    JITTOR_IMPORT_ERROR = exc
else:
    JITTOR_IMPORT_ERROR = None


def require_jittor() -> None:
    if jt is None or nn is None:
        raise RuntimeError(
            "Jittor import failed. Please install Jittor and ensure a compatible Python/C++ build environment. "
            f"Original error: {JITTOR_IMPORT_ERROR}"
        )


if nn is not None:

    class MLPScorer(nn.Module):
        """A small scorer that maps 1537-dim features to one relevance score."""

        def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, 1)

        def execute(self, features):
            x = self.fc1(features)
            x = self.relu(x)
            x = self.dropout(x)
            return self.fc2(x).squeeze(-1)

else:

    class MLPScorer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_jittor()


def pairwise_ranking_loss(score_pos, score_neg):
    require_jittor()
    return -jt.log(jt.sigmoid(score_pos - score_neg) + 1e-8).mean()
