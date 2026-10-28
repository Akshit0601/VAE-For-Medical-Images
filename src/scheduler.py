import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, _LRScheduler


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    warmup_start_factor: float = 0.1,
    min_lr: float = 1e-6,
) -> _LRScheduler:
    """
    Creates a chained Warmup + Cosine Annealing learning rate scheduler:
    1. Epochs 1 to warmup_epochs: LinearLR warming up from (base_lr * warmup_start_factor) to base_lr.
    2. Epochs warmup_epochs+1 to total_epochs: CosineAnnealingLR decaying from base_lr to min_lr.
    """
    if warmup_epochs >= total_epochs:
        raise ValueError(f"warmup_epochs ({warmup_epochs}) must be strictly less than total_epochs ({total_epochs})")

    # Linear warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Cosine decay down to min_lr
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    # Combine sequentially
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    return scheduler
