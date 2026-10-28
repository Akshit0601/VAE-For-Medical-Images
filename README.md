# Convolutional VAE for Chest X-Ray Medical Image Generation

A PyTorch implementation of a Convolutional Variational Autoencoder (Conv-VAE) trained on the Kaggle [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset. Includes on-demand dataset caching, AdamW optimization with Warmup Cosine learning rate scheduling, TensorBoard visualization, model checkpointing, and evaluation using Fréchet Inception Distance (FID) and Inception Score (IS).

---

## 1. Environment Setup

### Prerequisites
* Python 3.12+
* NVIDIA GPU with CUDA drivers (e.g., RTX 4050)
* Kaggle account with API credentials

### Step-by-Step Setup

1. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Installs PyTorch with CUDA acceleration, `tensorboard`, `kaggle`, `pytorch-ignite`, `torchvision`, `pillow`, and supporting numerical libraries.*

3. **Configure Kaggle Credentials** *(Required for automatic dataset download)*:
   ```bash
   kaggle auth login
   ```

---

## 2. Workspace Structure

```
vae_setup/
├── .gitignore            # Git ignore rules
├── requirements.txt      # Project dependencies
├── config.py             # Hyperparameter & training configuration
├── train.py              # Training loop, checkpointing & TensorBoard logging
├── evaluate.py           # FID & Inception Score evaluation script
├── verify_env.py         # Environment & CUDA smoke test
├── README.md             # Project documentation
│
├── src/
│   ├── __init__.py       # Package exports
│   ├── model.py          # Conv-VAE architecture (Encoder, Decoder, VAE)
│   ├── dataset.py        # Kaggle dataset downloader, caching & DataLoaders
│   └── scheduler.py      # Warmup Cosine learning rate scheduler
│
├── data/                 # Cached dataset directory
├── checkpoints/          # Saved model checkpoints
└── runs/                 # TensorBoard event logs
```

---

## 3. Source Modules (`src/`)

| Script | Purpose & Usecase |
| :--- | :--- |
| **`src/model.py`** | Implements modular `nn.Module` classes: <br>• **`Encoder`**: 5-stage downsampling (`Conv2d` $\to$ `BatchNorm2d` $\to$ `LeakyReLU`) projecting to $\mu$ and $\log \sigma^2$. <br>• **`Decoder`**: Feature projection, 5-stage upsampling (`ConvTranspose2d` $\to$ `BatchNorm2d` $\to$ `LeakyReLU`), and final `Tanh()` activation. <br>• **`VAE`**: Connects Encoder and Decoder with the Gaussian reparameterization trick ($z = \mu + \sigma \odot \epsilon$), sampling methods, and total loss computation ($\text{MSE} + \beta \cdot D_{\text{KL}}$). |
| **`src/dataset.py`** | Data handling and caching: <br>• **`ensure_dataset()`**: Checks if `data/chest_xray` exists. If not, automatically downloads and extracts the dataset from Kaggle. <br>• **Caching**: The raw images are cached in `data/`; subsequent runs load directly from disk without re-downloading. <br>• **`get_dataloaders()`**: Builds PyTorch `DataLoader` instances for `train` (5,216 images), `val` (16 images), and `test` (624 images) splits, converting images to 3-channel RGB and normalizing to $[-1, 1]$. |
| **`src/scheduler.py`** | Implements `get_warmup_cosine_scheduler()`: <br>• Linearly ramps the learning rate during the first few warmup epochs to prevent early posterior collapse. <br>• Applies smooth Cosine Annealing decay down to `min_lr` over the remaining epochs for fine-grained feature sharpening. |

---

## 4. Configuration Parameters (`config.py`)

All hyperparameters and pipeline options are centralized in `config.py`:

```python
@dataclass
class VAEConfig:
    # --- Dataset Settings ---
    dataset_name: str = "chest-xray"
    kaggle_dataset_slug: str = "paultimothymooney/chest-xray-pneumonia"
    data_dir: str = "data"                   # Cached dataset location
    img_size: int = 64                       # Target image resolution (64x64)
    in_channels: int = 3                     # RGB channels matching Tanh output
    num_workers: int = 2                     # Background data loader workers

    # --- Model Architecture ---
    hidden_dims: List[int] = [32, 64, 128, 256, 512]  # Channel sizes per layer
    latent_dim: int = 1024                   # Dimension of bottleneck latent code z

    # --- Training Hyperparameters ---
    batch_size: int = 16                     # Batch size for training & evaluation
    num_epochs: int = 80                     # Total training epochs
    warmup_epochs: int = 5                   # Linear learning rate warmup epochs
    base_lr: float = 5e-4                    # Peak learning rate for AdamW
    min_lr: float = 1e-6                     # Minimum learning rate at end of cosine decay
    weight_decay: float = 1e-2               # Decoupled weight decay for AdamW
    kld_weight: float = 0.00025              # Beta weight balancing Reconstruction vs KL loss

    # --- Logging & Checkpoints ---
    log_dir: str = "runs/chest_xray_vae/batch_16"       # TensorBoard directory
    checkpoint_dir: str = "checkpoints/batch_16"        # Checkpoint directory
    sample_interval_epochs: int = 2          # Log visual image grids every N epochs
```

---

## 5. Training (`train.py`)

To start model training:
```bash
source .venv/bin/activate
python train.py
```

### What `train.py` Does:
1. **Dataset Caching**: Checks `data/chest_xray`. If missing, downloads and unzips from Kaggle automatically. If already present, skips downloading and uses the local cache.
2. **Optimization**: Trains using `AdamW` and the `Warmup Cosine` scheduler on GPU (`cuda`).
3. **TensorBoard Logging**: Generates live logs in `runs/`:
   * **Scalars**: `Loss/Train_Total`, `Loss/Train_Reconstruction`, `Loss/Train_KLD`, validation losses, and `Learning_Rate`.
   * **Images**: Visual comparison grids showing real validation images side-by-side with VAE reconstructions, and random synthetic images sampled from $\mathcal{N}(0, I)$.
4. **Checkpointing**:
   * Automatically saves the best model based on validation loss to `checkpoints/batch_16/best_vae.pt`.
   * Saves the most recent epoch weights to `checkpoints/batch_16/latest_vae.pt`.

### Live Monitoring:
In another terminal, launch TensorBoard:
```bash
source .venv/bin/activate
tensorboard --logdir runs/chest_xray_vae --port 6006
```
Open **`http://localhost:6006`** in your browser.

---

## 6. Evaluation (`evaluate.py`)

Computes **Fréchet Inception Distance (FID)** and **Inception Score (IS)** using `pytorch-ignite`:

1. Set your desired checkpoint at the top of [`evaluate.py`](file:///home/akshit/vae_setup/evaluate.py#L23):
   ```python
   CKPT_PATH = "checkpoints/batch_16/best_vae.pt"
   ```
2. Run evaluation:
   ```bash
   python evaluate.py
   ```
3. **Outputs**:
   * Prints FID score (lower is better) and Inception Score (higher is better) evaluated against the 624 real test images.
   * Saves a 16-sample grid of generated synthetic X-rays to `outputs/generated_samples_for_report.png`.
