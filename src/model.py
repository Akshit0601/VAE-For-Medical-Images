from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional Encoder for VAE.
    Downsamples the input image through a sequence of Conv2d -> BatchNorm2d -> LeakyReLU blocks,
    then projects the flattened feature map to latent Gaussian parameters (mu, log_var).
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 1024,
        hidden_dims: Optional[List[int]] = None,
        img_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build convolutional downsampling blocks
        modules = []
        curr_channels = in_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        curr_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            curr_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Calculate spatial feature size at the end of encoder
        # Each layer with kernel=3, stride=2, padding=1 halves spatial dimension:
        # e.g., 64 -> 32 -> 16 -> 8 -> 4 -> 2 for 5 layers
        self.spatial_dim = img_size // (2 ** len(self.hidden_dims))
        if self.spatial_dim < 1:
            raise ValueError(
                f"img_size {img_size} is too small for {len(self.hidden_dims)} downsampling layers."
            )
        self.flattened_dim = self.hidden_dims[-1] * (self.spatial_dim ** 2)

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_var = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        latent = torch.flatten(latent, start_dim=1)
        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)
        return mu, log_var


class Decoder(nn.Module):
    """
    Convolutional Decoder for VAE.
    Projects the sampled latent vector z back to spatial feature maps and upsamples
    using ConvTranspose2d -> BatchNorm2d -> LeakyReLU, finishing with Conv2d -> Tanh().
    """
    def __init__(
        self,
        latent_dim: int = 1024,
        out_channels: int = 3,
        hidden_dims: Optional[List[int]] = None,
        spatial_dim: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Reverse hidden dimensions for decoder (e.g. [512, 256, 128, 64, 32])
        rev_hidden_dims = hidden_dims.copy()
        rev_hidden_dims.reverse()
        self.rev_hidden_dims = rev_hidden_dims

        self.decoder_input = nn.Linear(
            latent_dim, self.rev_hidden_dims[0] * (spatial_dim ** 2)
        )

        modules = []
        for i in range(len(self.rev_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.rev_hidden_dims[i],
                        self.rev_hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(self.rev_hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder_body = nn.Sequential(*modules)

        # Final upsampling block bringing feature map back to target image dimensions
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                self.rev_hidden_dims[-1],
                self.rev_hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(self.rev_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.rev_hidden_dims[-1],
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(
            -1, self.rev_hidden_dims[0], self.spatial_dim, self.spatial_dim
        )
        result = self.decoder_body(result)
        result = self.final_layer(result)
        return result


class VAE(nn.Module):
    """
    Standard Variational Autoencoder combining Encoder and Decoder modules.
    Includes reparameterization trick, sampling, reconstruction, and loss calculation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 1024,
        hidden_dims: Optional[List[int]] = None,
        img_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            img_size=img_size,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            hidden_dims=hidden_dims,
            spatial_dim=self.encoder.spatial_dim,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps, where eps ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

    def loss_function(
        self,
        recons: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kld_weight: float = 0.00025,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the VAE loss function:
        Total Loss = Reconstruction Loss (MSE) + kld_weight * KL Divergence
        """
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
            dim=0,
        )
        total_loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": total_loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: torch.device) -> torch.Tensor:
        """
        Samples latent vectors from the prior N(0, I) and generates reconstructions.
        """
        z = torch.randn(num_samples, self.latent_dim, device=current_device)
        samples = self.decoder(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image.
        """
        recons, _, _ = self.forward(x)
        return recons
