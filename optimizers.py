"""
Utility functions to create optimizers and learning rate schedulers for training neural networks in PyTorch.
"""

from typing import Iterable, Optional
import torch
from torch.optim import Optimizer


def make_optimizer(
    params: Iterable,
    name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    momentum: float,
    eps: float,
) -> Optimizer:
    """
    Returns a PyTorch optimizer based on the given name and hyperparameters.
    """
    n = name.lower()
    if n == "adam":
        return torch.optim.Adam(
            params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
    if n == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
    if n == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )
    if n == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(
    opt: Optimizer,
    name: Optional[str],  # step|cosine|plateau|None
    step_size: int,
    gamma: float,
    T_max: int,
):
    """
    Returns a learning rate scheduler based on the given name and hyperparameters.
    """
    if name is None:
        return None
    n = name.lower()
    if n == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
    if n == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=gamma, patience=step_size
        )
    raise ValueError(f"Unknown scheduler: {name}")
