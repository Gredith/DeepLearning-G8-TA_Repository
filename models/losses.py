"""Binary classification losses."""

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss on logits; alpha=None makes gamma=0 equal BCE."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        loss = (1 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
