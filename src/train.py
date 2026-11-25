"""Clean, readable training loop for Phase 1 unconditional GANs."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_unconditional_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    latent_dim: int = 64,
    lr: float = 2e-4,
    lr_g: float | None = None,
    lr_d: float | None = None,
    betas: tuple[float, float] = (0.5, 0.999),
    real_label_value: float = 0.9,
    fixed_noise: torch.Tensor | None = None,
    snapshot_epochs: Iterable[int] = (1, 5, 10, 15, 25),
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train an unconditional GAN using Binary Cross-Entropy with logits.

    Args:
        generator: Generator network mapping noise z -> fake images.
        discriminator: Discriminator network mapping images -> real/fake logits.
        dataloader: Training DataLoader yielding (images, labels).
        device: Compute device (cuda / mps / cpu).
        epochs: Number of full passes over the dataset.
        latent_dim: Size of the Gaussian noise vector z.
        lr: Default Adam learning rate for both Generator and Discriminator.
        lr_g: Optional override learning rate for the Generator (TTUR).
        lr_d: Optional override learning rate for the Discriminator (TTUR).
        betas: Adam beta hyperparameters (0.5, 0.999 is standard for GANs).
        real_label_value: Target for real images in D's loss. Using 0.9 instead
            of 1.0 applies one-sided label smoothing to keep D from overpowering G.
        fixed_noise: Optional fixed batch of noise vectors used to visualize
            progress on the exact same inputs across epochs.
        snapshot_epochs: Epoch numbers (1-indexed) at which to save generated
            samples from `fixed_noise`.
        show_progress: Display a tqdm progress bar during training.

    Returns:
        Dictionary containing per-epoch loss/score histories and image snapshots.
    """
    generator.to(device)
    discriminator.to(device)

    gen_lr = lr if lr_g is None else lr_g
    disc_lr = lr if lr_d is None else lr_d

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=gen_lr, betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, betas=betas)

    if fixed_noise is None:
        fixed_noise = torch.randn(32, latent_dim, device=device)
    else:
        fixed_noise = fixed_noise.to(device)

    snapshot_set = set(snapshot_epochs)
    history: dict[str, Any] = {
        "g_losses": [],
        "d_losses": [],
        "d_real_scores": [],
        "d_fake_scores": [],
        "snapshots": {},
    }

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        running_d_real = 0.0
        running_d_fake = 0.0
        num_batches = 0

        iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch:02d}/{epochs}",
            leave=False,
            disable=not show_progress,
        )

        for real_images, _ in iterator:
            batch_size = real_images.size(0)
            real_images = real_images.to(device, non_blocking=True)

            # Target tensors for BCEWithLogitsLoss
            real_targets = torch.full((batch_size, 1), real_label_value, device=device)
            fake_targets = torch.zeros((batch_size, 1), device=device)
            gen_targets = torch.ones((batch_size, 1), device=device)

            # -----------------------------------------------------------------
            # Step 1: Train Discriminator to distinguish real vs. fake images
            # -----------------------------------------------------------------
            opt_d.zero_grad()

            real_logits = discriminator(real_images)
            loss_d_real = criterion(real_logits, real_targets)

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            # Detach fake_images so gradients do not flow into G during D's step
            fake_logits = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_logits, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # -----------------------------------------------------------------
            # Step 2: Train Generator to fool the updated Discriminator
            # -----------------------------------------------------------------
            opt_g.zero_grad()

            # Pass the freshly generated images through D (without detach)
            gen_logits = discriminator(fake_images)
            loss_g = criterion(gen_logits, gen_targets)
            loss_g.backward()
            opt_g.step()

            # Track metrics for logging
            running_g_loss += loss_g.item()
            running_d_loss += loss_d.item()
            running_d_real += torch.sigmoid(real_logits).mean().item()
            running_d_fake += torch.sigmoid(fake_logits).mean().item()
            num_batches += 1

        avg_g_loss = running_g_loss / num_batches
        avg_d_loss = running_d_loss / num_batches
        avg_d_real = running_d_real / num_batches
        avg_d_fake = running_d_fake / num_batches

        history["g_losses"].append(avg_g_loss)
        history["d_losses"].append(avg_d_loss)
        history["d_real_scores"].append(avg_d_real)
        history["d_fake_scores"].append(avg_d_fake)

        if epoch in snapshot_set or epoch == epochs:
            generator.eval()
            with torch.no_grad():
                history["snapshots"][epoch] = generator(fixed_noise).cpu()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"D loss: {avg_d_loss:.4f} | G loss: {avg_g_loss:.4f} | "
            f"D(real): {avg_d_real:.3f} | D(fake): {avg_d_fake:.3f}"
        )

    return history
