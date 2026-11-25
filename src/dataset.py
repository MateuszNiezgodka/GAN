"""Data loading utilities for the EMNIST Letters dataset.

Handles the two well-known quirks of `torchvision.datasets.EMNIST(split="letters")`:
1. Images are stored transposed (rotated 90 degrees and flipped horizontally).
2. Labels are 1-indexed (1..26 for A..Z) instead of 0-indexed (0..25).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

LETTER_CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def label_to_letter(label: int) -> str:
    """Convert a 0-indexed class ID (0..25) to its uppercase letter ('A'..'Z')."""
    if not 0 <= label < len(LETTER_CLASSES):
        raise ValueError(f"Expected label in 0..25, got {label}.")
    return LETTER_CLASSES[label]


def letter_to_label(letter: str) -> int:
    """Convert a single letter ('A'..'Z', case-insensitive) to its 0-indexed class ID."""
    cleaned = letter.strip().upper()
    if len(cleaned) != 1 or cleaned not in LETTER_CLASSES:
        raise ValueError(f"Expected a single letter A-Z, got {letter!r}.")
    return LETTER_CLASSES.index(cleaned)


def fix_emnist_orientation(image: torch.Tensor) -> torch.Tensor:
    """Transpose a (C, H, W) tensor so EMNIST letters appear upright."""
    return image.transpose(1, 2)


def shift_emnist_label(label: int) -> int:
    """Shift EMNIST Letters labels from 1..26 to 0..25."""
    return label - 1


def get_emnist_transforms(
    fix_orientation: bool = True,
    normalize: bool = True,
) -> transforms.Compose:
    """Build the image transform pipeline for EMNIST Letters.

    Args:
        fix_orientation: Transpose images so letters are upright. Set to False only
            when demonstrating the raw EMNIST orientation quirk.
        normalize: Scale pixel values from [0, 1] to [-1, 1] to match Generator Tanh().
    """
    steps: list = [transforms.ToTensor()]

    if fix_orientation:
        steps.append(transforms.Lambda(fix_emnist_orientation))

    if normalize:
        steps.append(transforms.Normalize((0.5,), (0.5,)))

    return transforms.Compose(steps)


def get_emnist_dataset(
    root: str | Path = "data",
    train: bool = True,
    download: bool = True,
    fix_orientation: bool = True,
    normalize: bool = True,
    allowed_labels: Iterable[int] | None = None,
    in_memory: bool = True,
) -> Dataset:
    """Load the EMNIST Letters dataset with fixed orientation and 0-indexed labels.

    Args:
        root: Directory where the dataset will be downloaded/cached.
        train: Load the training split (124,800 images) if True, else test (20,800 images).
        download: Automatically download EMNIST if not already present locally.
        fix_orientation: Fix the default rotated/flipped EMNIST image orientation.
        normalize: Normalize images to [-1, 1].
        allowed_labels: Optional collection of 0-indexed class labels to keep
            (used in Phase 3 to separate seen vs. unseen letters).
        in_memory: Pre-convert all 28x28 tensors into a TensorDataset in RAM (~390 MB).
            This avoids slow per-image PIL conversions and multiprocessing overhead.
    """
    raw_dataset = datasets.EMNIST(
        root=str(root),
        split="letters",
        train=train,
        download=download,
        transform=get_emnist_transforms(
            fix_orientation=fix_orientation,
            normalize=normalize,
        ),
        target_transform=shift_emnist_label,
    )

    if in_memory:
        # Vectorized equivalent of ToTensor -> fix_orientation -> Normalize((0.5,), (0.5,))
        images = raw_dataset.data.float().div(255.0).unsqueeze(1)  # (N, 1, 28, 28)
        if fix_orientation:
            images = images.transpose(2, 3).contiguous()
        if normalize:
            images = images.mul(2.0).sub(1.0)

        labels = raw_dataset.targets.long() - 1  # Shift 1..26 -> 0..25

        if allowed_labels is not None:
            allowed_set = set(allowed_labels)
            mask = torch.tensor([int(y) in allowed_set for y in labels.tolist()], dtype=torch.bool)
            images = images[mask]
            labels = labels[mask]

        return TensorDataset(images, labels)

    if allowed_labels is None:
        return raw_dataset

    # Filter indices using the raw 1-indexed targets stored on the dataset object.
    allowed_raw = {label + 1 for label in allowed_labels}
    indices = [
        idx for idx, raw_label in enumerate(raw_dataset.targets.tolist())
        if raw_label in allowed_raw
    ]
    return Subset(raw_dataset, indices)


def get_emnist_dataloader(
    batch_size: int = 128,
    train: bool = True,
    shuffle: bool | None = None,
    root: str | Path = "data",
    num_workers: int = 0,
    download: bool = True,
    fix_orientation: bool = True,
    normalize: bool = True,
    allowed_labels: Iterable[int] | None = None,
    pin_memory: bool | None = None,
    in_memory: bool = True,
) -> DataLoader:
    """Create a fast DataLoader for EMNIST Letters."""
    dataset = get_emnist_dataset(
        root=root,
        train=train,
        download=download,
        fix_orientation=fix_orientation,
        normalize=normalize,
        allowed_labels=allowed_labels,
        in_memory=in_memory,
    )

    if shuffle is None:
        shuffle = train

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Drop the final partial batch during training so BatchNorm never sees size 1.
        drop_last=train,
    )
