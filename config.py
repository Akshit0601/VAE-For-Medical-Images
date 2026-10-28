from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class VAEConfig:
    # --- Dataset settings ---
    dataset_name: str = "chest-xray"
    kaggle_dataset_slug: str = "paultimothymooney/chest-xray-pneumonia"
    data_dir: str = "data"
    img_size: int = 64
    in_channels: int = 3
    num_workers: int = 2

    # --- Model Architecture ---
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    latent_dim: int = 1024

    # --- Training Hyperparameters ---
    batch_size: int = 16
    num_epochs: int = 80       # Increased to 80 epochs (~26,080 steps)
    warmup_epochs: int = 5
    base_lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    kld_weight: float = 0.00025  # normalized beta weight

    # --- Logging & Checkpoints ---
    log_dir: str = "runs/chest_xray_vae/batch_16"
    checkpoint_dir: str = "checkpoints/batch_16"
    sample_interval_epochs: int = 2
    save_best_only: bool = True

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Default configuration instance
default_config = VAEConfig()
