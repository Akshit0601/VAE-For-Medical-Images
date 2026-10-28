import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def get_transforms(img_size: int = 64) -> transforms.Compose:
    """
    Returns image transformation pipeline:
    - Converts grayscale / single-channel images to 3 RGB channels.
    - Resizes to target resolution (e.g. 64x64).
    - Converts PIL to FloatTensor [0, 1].
    - Normalizes to [-1, 1] to match the VAE decoder's Tanh() output.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def find_data_split(base_dir: str, split_name: str) -> Optional[str]:
    """
    Recursively finds the subfolder for train/val/test splits,
    handling nested zip extractions like data/chest_xray/chest_xray/train.
    """
    for root, dirs, _ in os.walk(base_dir):
        if split_name.lower() in [d.lower() for d in dirs]:
            for d in dirs:
                if d.lower() == split_name.lower():
                    candidate = os.path.join(root, d)
                    # Verify it has class folders (NORMAL, PNEUMONIA)
                    subdirs = [s for s in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, s))]
                    if subdirs:
                        return candidate
    return None


def ensure_dataset(data_dir: str = "data", kaggle_slug: str = "paultimothymooney/chest-xray-pneumonia") -> str:
    """
    Checks if the dataset exists locally in data_dir.
    If not, downloads and unzips it on-demand via the Kaggle API.
    """
    train_dir = find_data_split(data_dir, "train")
    if train_dir is not None:
        return data_dir

    print(f"\n[Data Pipeline] Dataset not found in '{data_dir}'. Attempting download via Kaggle API: '{kaggle_slug}'...")

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(kaggle_slug, path=data_dir, unzip=True)
        print(f"[Data Pipeline] Successfully downloaded and extracted '{kaggle_slug}' into '{data_dir}'.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to automatically download Kaggle dataset '{kaggle_slug}'.\n"
            f"Error details: {e}\n\n"
            f"Please ensure your Kaggle credentials are set up. Options:\n"
            f"  1. Run 'kaggle auth login' in your terminal.\n"
            f"  2. Place your 'kaggle.json' in '~/.kaggle/kaggle.json' (chmod 600 ~/.kaggle/kaggle.json).\n"
            f"  3. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        ) from e

    train_dir = find_data_split(data_dir, "train")
    if train_dir is None:
        raise FileNotFoundError(f"Extracted dataset in '{data_dir}', but could not locate 'train' folder.")

    return data_dir


def get_dataloaders(
    data_dir: str = "data",
    kaggle_slug: str = "paultimothymooney/chest-xray-pneumonia",
    img_size: int = 64,
    batch_size: int = 8,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, validation, and test splits.
    """
    if download:
        ensure_dataset(data_dir=data_dir, kaggle_slug=kaggle_slug)

    train_path = find_data_split(data_dir, "train")
    val_path = find_data_split(data_dir, "val")
    test_path = find_data_split(data_dir, "test")

    if not train_path:
        raise FileNotFoundError(f"Could not find 'train' split folder in '{data_dir}'.")

    transform = get_transforms(img_size=img_size)

    train_dataset = ImageFolder(root=train_path, transform=transform)
    val_dataset = ImageFolder(root=val_path, transform=transform) if val_path else None
    test_dataset = ImageFolder(root=test_path, transform=transform) if test_path else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_dataset is not None
        else train_loader
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_dataset is not None
        else val_loader
    )

    return train_loader, val_loader, test_loader
