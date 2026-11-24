"""GAN model architectures for Phase 1 (Basic Unconditional GAN).

Includes:
- MLPGenerator / MLPDiscriminator: Simple fully connected baseline (Trial 1).
- DCGANGenerator / DCGANDiscriminator: Compact convolutional GAN for 28x28 images (Trial 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def init_gan_weights(module: nn.Module) -> None:
    """Initialize Conv, Linear, and BatchNorm layers following the DCGAN paper."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Trial 1: Simple Fully Connected (MLP) GAN Baseline
# ---------------------------------------------------------------------------


class MLPGenerator(nn.Module):
    """Fully connected generator mapping random noise z -> (1, 28, 28) image."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 28 * 28),
            nn.Tanh(),  # Output pixels in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat_images = self.net(z)
        return flat_images.view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    """Fully connected discriminator classifying (1, 28, 28) images as real/fake logits."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # Raw logits for BCEWithLogitsLoss
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


# ---------------------------------------------------------------------------
# Trial 2: Compact Deep Convolutional GAN (DCGAN) for 28x28 Images
# ---------------------------------------------------------------------------


class DCGANGenerator(nn.Module):
    """Convolutional generator mapping z (B, latent_dim) -> image (B, 1, 28, 28).

    Spatial progression:
        (B, latent_dim) -> (B, 128, 7, 7) -> (B, 64, 14, 14) -> (B, 1, 28, 28)
    """

    def __init__(self, latent_dim: int = 64, base_channels: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.start_channels = base_channels * 2

        self.project = nn.Linear(latent_dim, self.start_channels * 7 * 7, bias=False)
        self.upsample = nn.Sequential(
            # (B, 128, 7, 7)
            nn.BatchNorm2d(self.start_channels),
            nn.ReLU(inplace=True),
            # (B, 128, 7, 7) -> (B, 64, 14, 14)
            nn.ConvTranspose2d(
                self.start_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # (B, 64, 14, 14) -> (B, 1, 28, 28)
            nn.ConvTranspose2d(
                base_channels,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),  # Output pixels in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(-1, self.start_channels, 7, 7)
        return self.upsample(x)


class DCGANDiscriminator(nn.Module):
    """Convolutional discriminator mapping image (B, 1, 28, 28) -> real/fake logit (B, 1).

    Spatial progression:
        (B, 1, 28, 28) -> (B, 64, 14, 14) -> (B, 128, 7, 7) -> (B, 1)
    """

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        hidden_channels = base_channels * 2

        self.features = nn.Sequential(
            # (B, 1, 28, 28) -> (B, 64, 14, 14)
            # Note: No BatchNorm on the first discriminator layer (DCGAN convention).
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 64, 14, 14) -> (B, 128, 7, 7)
            nn.Conv2d(
                base_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 7 * 7, 1),  # Raw logits for BCEWithLogitsLoss
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)
