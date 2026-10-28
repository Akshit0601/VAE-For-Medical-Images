#!/usr/bin/env python3

import os
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from ignite.metrics import FID, InceptionScore
from tqdm import tqdm

from src.model import VAE
from src.dataset import get_dataloaders
from config import default_config

# ==============================================================================
# Model Checkpoint & Evaluation Parameters
# Change CKPT_PATH to evaluate any specific saved checkpoint:
# ==============================================================================
CKPT_PATH = "/home/akshit/vae_setup/checkpoints/batch_16/best_vae.pt"
BATCH_SIZE = 16
NUM_SAMPLES = 624   # Number of real/generated images to compare (624 test images available)
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================================================


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Converts [-1, 1] normalized images to [0, 1]."""
    return torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Loads VAE model architecture and state dictionary from checkpoint."""
    if not os.path.exists(ckpt_path):
        # Fallback check inside checkpoints/batch_8 or batch_16 if root doesn't exist
        for alt_path in ["checkpoints/batch_8/best_vae.pt", "checkpoints/batch_16/best_vae.pt"]:
            if os.path.exists(alt_path):
                print(f"[Warning] '{ckpt_path}' not found, falling back to '{alt_path}'")
                ckpt_path = alt_path
                break

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config", default_config)

    model = VAE(
        in_channels=cfg.in_channels,
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        img_size=cfg.img_size,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "N/A")
    val_loss = checkpoint.get("val_loss", "N/A")
    print(f"Checkpoint successfully loaded (Epoch: {epoch}, Validation Loss: {val_loss})")

    return model, checkpoint


def main():
    print("=" * 65)
    print("       Conv-VAE Evaluation: Fréchet Inception Distance & Inception Score")
    print("=" * 65)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # 1. Load Model Checkpoint
    model, checkpoint_info = load_model_from_checkpoint(CKPT_PATH, device)

    # 2. Prepare Test Data
    print("\n[1/3] Loading real test dataset...")
    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        kaggle_slug=default_config.kaggle_dataset_slug,
        img_size=default_config.img_size,
        batch_size=BATCH_SIZE,
        num_workers=2,
        download=False,
    )
    print(f"Test dataset contains {len(test_loader.dataset)} images.")

    # 3. Initialize PyTorch-Ignite Metrics
    print("\n[2/3] Initializing PyTorch-Ignite FID and InceptionScore metrics...")
    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device)

    # 4. Generate & Compare Samples
    print(f"\n[3/3] Evaluating on {NUM_SAMPLES} real and synthetic samples...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    samples_processed = 0
    test_iter = iter(test_loader)

    sample_grid_fakes = []

    pbar = tqdm(total=NUM_SAMPLES, desc="Computing FID & IS")
    while samples_processed < NUM_SAMPLES:
        # Get real batch
        try:
            real_batch, _ = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            real_batch, _ = next(test_iter)

        curr_bs = min(len(real_batch), NUM_SAMPLES - samples_processed)
        real_batch = real_batch[:curr_bs].to(device)

        # Scale real images from [-1, 1] to [0, 1]
        real_images = denormalize(real_batch)

        # Generate fake images from prior N(0, I)
        with torch.no_grad():
            fake_batch = model.sample(num_samples=curr_bs, current_device=device)
            fake_images = denormalize(fake_batch)

        # Collect first 16 samples for visual output
        if len(sample_grid_fakes) < 16:
            sample_grid_fakes.append(fake_images.cpu())

        # Inception-v3 expects 299x299 resolution
        real_299 = F.interpolate(real_images, size=(299, 299), mode="bilinear", align_corners=False)
        fake_299 = F.interpolate(fake_images, size=(299, 299), mode="bilinear", align_corners=False)

        # Update Ignite metrics
        fid_metric.update((fake_299, real_299))
        is_metric.update(fake_299)

        samples_processed += curr_bs
        pbar.update(curr_bs)

    pbar.close()

    # 5. Compute Final Metric Scores
    print("\nComputing final metric values...")
    fid_score = fid_metric.compute()
    is_score = is_metric.compute()

    # 6. Save Sample Generated Images for Report
    if sample_grid_fakes:
        sample_tensor = torch.cat(sample_grid_fakes, dim=0)[:16]
        grid = make_grid(sample_tensor, nrow=4, normalize=False)
        output_grid_path = os.path.join(OUTPUT_DIR, "generated_samples_for_report.png")
        save_image(grid, output_grid_path)
        print(f"Saved 16 synthetic sample X-rays to: {output_grid_path}")

    # 7. Print Report Summary
    print("\n" + "=" * 65)
    print("                     EVALUATION RESULTS")
    print("=" * 65)
    print(f"Checkpoint Evaluated    : {CKPT_PATH}")
    print(f"Samples Evaluated       : {samples_processed}")
    print("-" * 65)
    print(f"Fréchet Inception Dist  : {fid_score:.4f}  (lower is better)")
    print(f"Inception Score (IS)    : {is_score:.4f}  (higher is better)")
    print("=" * 65)


if __name__ == "__main__":
    main()
