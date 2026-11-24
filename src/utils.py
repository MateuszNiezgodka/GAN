"""Utility helpers for reproducibility, plotting, and saving/loading checkpoints."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.dataset import label_to_letter


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select the best available compute device (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] back to [0, 1] for matplotlib display."""
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)


def show_image_grid(
    images: torch.Tensor,
    labels: Sequence[int] | torch.Tensor | None = None,
    nrow: int = 8,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    is_normalized: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Display a batch of grayscale (N, 1, 28, 28) images in a clean grid."""
    imgs = images.detach().cpu()
    if is_normalized:
        imgs = denormalize(imgs)

    num_images = imgs.shape[0]
    ncols = min(nrow, num_images)
    nrows = math.ceil(num_images / ncols)

    if figsize is None:
        figsize = (ncols * 1.3, nrows * 1.4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).reshape(-1)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()

    for idx, ax in enumerate(axes_flat):
        ax.axis("off")
        if idx >= num_images:
            continue

        ax.imshow(imgs[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        if labels is not None:
            ax.set_title(label_to_letter(int(labels[idx])), fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_training_losses(
    g_losses: Sequence[float],
    d_losses: Sequence[float],
    d_real_scores: Sequence[float] | None = None,
    d_fake_scores: Sequence[float] | None = None,
    title: str = "GAN Training Dynamics",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot Generator/Discriminator losses (and optionally D's average probabilities)."""
    epochs = range(1, len(g_losses) + 1)
    show_scores = d_real_scores is not None and d_fake_scores is not None

    ncols = 2 if show_scores else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    # Left plot: BCE losses
    axes[0].plot(epochs, g_losses, label="Generator Loss", linewidth=2)
    axes[0].plot(epochs, d_losses, label="Discriminator Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Losses per Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Right plot: Discriminator confidence D(x) and D(G(z))
    if show_scores:
        axes[1].plot(epochs, d_real_scores, label="Mean D(real)", linewidth=2)
        axes[1].plot(epochs, d_fake_scores, label="Mean D(fake)", linewidth=2)
        axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Equilibrium (0.5)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Discriminator Output Probability")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Discriminator Confidence")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    epoch: int,
    history: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Path:
    """Save trained model weights and training history to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "history": history,
        "config": config or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load saved weights into model instances and return the full checkpoint dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    if discriminator is not None and "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    return checkpoint
