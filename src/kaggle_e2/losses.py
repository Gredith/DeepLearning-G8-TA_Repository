"""Pérdida focal binaria (Lin et al., 2017) sobre logits.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Decisión de diseño (ver DECISIONES.md, D-07): por defecto alpha=None, es decir,
NO se pondera por clase. Así, la única diferencia con la línea base es gamma, y
con gamma=0 la pérdida es exactamente BCE: "una hipótesis, un cambio".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, logits, targets):
        logits, targets = logits.view(-1).float(), targets.view(-1).float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # = -log(p_t)
        p_t = torch.exp(-bce)
        loss = (1 - p_t) ** self.gamma * bce if self.gamma > 0 else bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(gamma: float, alpha: float | None = None):
    return BinaryFocalLoss(gamma=gamma, alpha=alpha)
