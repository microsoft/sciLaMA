"""
Reconstruction quality metrics using torchmetrics.
Pearson and Spearman correlation between ground-truth and reconstructed expression.
"""
import torch
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef


def _flatten_and_check(preds: torch.Tensor, target: torch.Tensor):
    """Flatten tensors to 1D for correlation. Handle edge cases."""
    preds = preds.detach().flatten().float()
    target = target.detach().flatten().float()
    if preds.numel() < 2:
        return preds, target
    return preds, target


def pearson_reconstruction(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation between predicted and target (reconstruction vs input).
    Returns scalar in [-1, 1]. Returns 0.0 if insufficient samples.
    """
    preds, target = _flatten_and_check(preds, target)
    if preds.numel() < 2:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    return pearson_corrcoef(preds, target)


def spearman_reconstruction(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spearman correlation between predicted and target (reconstruction vs input).
    Returns scalar in [-1, 1]. Returns 0.0 if insufficient samples.
    """
    preds, target = _flatten_and_check(preds, target)
    if preds.numel() < 2:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    return spearman_corrcoef(preds, target)
