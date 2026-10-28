#!/usr/bin/env python3
"""
Main Training Script for Convolutional VAE on Chest X-Ray Dataset.
All configuration parameters are defined directly in config.py.
"""

import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import default_config, VAEConfig
from src.model import VAE
from src.scheduler import get_warmup_cosine_scheduler
from src.dataset import get_dataloaders


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Converts [-1, 1] normalized images to [0, 1] for visual display."""
    return torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)


def train_one_epoch(
    model: VAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kld_weight: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for x, _ in pbar:
        x = x.to(device)
        optimizer.zero_grad()

        recons, mu, log_var = model(x)
        losses = model.loss_function(recons, x, mu, log_var, kld_weight=kld_weight)

        loss = losses["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += losses["Reconstruction_Loss"].item()
        total_kld += losses["KLD"].item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "recon": f"{losses['Reconstruction_Loss'].item():.4f}"})

    return {
        "loss": total_loss / max(num_batches, 1),
        "recon": total_recon / max(num_batches, 1),
        "kld": total_kld / max(num_batches, 1),
    }


def evaluate(
    model: VAE,
    val_loader: DataLoader,
    device: torch.device,
    kld_weight: float,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    num_batches = 0

    first_batch_inputs = None
    first_batch_recons = None

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            recons, mu, log_var = model(x)
            losses = model.loss_function(recons, x, mu, log_var, kld_weight=kld_weight)

            total_loss += losses["loss"].item()
            total_recon += losses["Reconstruction_Loss"].item()
            total_kld += losses["KLD"].item()
            num_batches += 1

            if first_batch_inputs is None:
                first_batch_inputs = x[:8]
                first_batch_recons = recons[:8]

    return (
        {
            "loss": total_loss / max(num_batches, 1),
            "recon": total_recon / max(num_batches, 1),
            "kld": total_kld / max(num_batches, 1),
        },
        first_batch_inputs,
        first_batch_recons,
    )


def train(cfg: VAEConfig = default_config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # DataLoaders (downloads on-demand if data/ is not yet present)
    print("[Data Pipeline] Preparing dataset and DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=cfg.data_dir,
        kaggle_slug=cfg.kaggle_dataset_slug,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        download=True,
    )

    # Model
    model = VAE(
        in_channels=cfg.in_channels,
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        img_size=cfg.img_size,
    ).to(device)

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
    )

    # Scheduler (Warmup Cosine)
    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.num_epochs,
        min_lr=cfg.min_lr,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.log_dir)
    print(f"TensorBoard logging to: {cfg.log_dir}")

    best_val_loss = float("inf")

    print("\nStarting Training:")
    print(f"  Dataset      : {cfg.dataset_name} ({cfg.kaggle_dataset_slug})")
    print(f"  Epochs       : {cfg.num_epochs}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Optimizer    : AdamW (lr={cfg.base_lr}, weight_decay={cfg.weight_decay})")
    print(f"  Scheduler    : Warmup Cosine (warmup={cfg.warmup_epochs} epochs)")
    print(f"  KLD Weight   : {cfg.kld_weight}")
    print("-" * 65)

    for epoch in range(1, cfg.num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg.kld_weight)

        # Validate
        val_metrics, val_orig, val_recons = evaluate(model, val_loader, device, cfg.kld_weight)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # TensorBoard Scalars
        writer.add_scalar("Loss/Train_Total", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/Train_Reconstruction", train_metrics["recon"], epoch)
        writer.add_scalar("Loss/Train_KLD", train_metrics["kld"], epoch)
        writer.add_scalar("Loss/Val_Total", val_metrics["loss"], epoch)
        writer.add_scalar("Loss/Val_Reconstruction", val_metrics["recon"], epoch)
        writer.add_scalar("Loss/Val_KLD", val_metrics["kld"], epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # TensorBoard Image Grids
        if epoch % cfg.sample_interval_epochs == 0 or epoch == cfg.num_epochs:
            if val_orig is not None and val_recons is not None:
                # Interleave originals and reconstructions side-by-side
                orig_disp = denormalize(val_orig.cpu())
                recons_disp = denormalize(val_recons.cpu())
                comparison = torch.cat([orig_disp, recons_disp], dim=0)
                grid_comp = make_grid(comparison, nrow=len(val_orig), normalize=False)
                writer.add_image("Visuals/Original_vs_Reconstruction", grid_comp, epoch)

            # Sample from prior N(0, I)
            with torch.no_grad():
                samples = model.sample(num_samples=8, current_device=device)
                grid_samples = make_grid(denormalize(samples.cpu()), nrow=4, normalize=False)
                writer.add_image("Visuals/Random_Prior_Samples", grid_samples, epoch)

        # Save Checkpoints
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(cfg.checkpoint_dir, "best_vae.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": cfg,
            }, best_path)

        latest_path = os.path.join(cfg.checkpoint_dir, "latest_vae.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "config": cfg,
        }, latest_path)

        print(
            f"Epoch [{epoch:02d}/{cfg.num_epochs:02d}] "
            f"Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon']:.4f}, KLD: {train_metrics['kld']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"LR: {current_lr:.6f} "
            f"{'(*Best*)' if is_best else ''}"
        )

    writer.close()
    print("-" * 65)
    print(f"Training completed successfully! Best model saved to: {os.path.join(cfg.checkpoint_dir, 'best_vae.pt')}")


if __name__ == "__main__":
    train(default_config)
