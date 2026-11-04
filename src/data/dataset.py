"""PyTorch Dataset and DataLoader factory for HAM10000.

Provides a custom Dataset class that returns (image, metadata, label)
tuples and a factory function that creates train/val/test DataLoaders
with appropriate augmentation and class-balanced sampling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.data.metadata import DIAGNOSIS_LABELS, MetadataEncoder


class HAM10000Dataset(Dataset):
    """HAM10000 dataset returning (image, metadata, label) tuples.

    Args:
        image_ids: List of image_id strings for this split.
        metadata_df: Full metadata DataFrame.
        image_dirs: List of directories containing the images.
        encoder: Fitted MetadataEncoder instance.
        transform: Torchvision transform to apply to images.
    """

    def __init__(
        self,
        image_ids: List[str],
        metadata_df: pd.DataFrame,
        image_dirs: List[str],
        encoder: MetadataEncoder,
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_ids = image_ids
        self.metadata_df = metadata_df.set_index("image_id")
        self.image_dirs = [Path(d) for d in image_dirs]
        self.encoder = encoder
        self.transform = transform

        # Pre-encode all metadata
        self._meta_cache: Dict[str, torch.Tensor] = {}
        self._label_cache: Dict[str, int] = {}
        for img_id in self.image_ids:
            row = self.metadata_df.loc[img_id]
            self._meta_cache[img_id] = encoder.encode(row)
            self._label_cache[img_id] = DIAGNOSIS_LABELS[row["dx"]]

    def _find_image(self, image_id: str) -> Path:
        """Locate an image file across the image directories."""
        for d in self.image_dirs:
            path = d / f"{image_id}.jpg"
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Image {image_id}.jpg not found in {self.image_dirs}"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image_id = self.image_ids[idx]

        # Load and transform image
        img_path = self._find_image(image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Retrieve cached metadata and label
        metadata = self._meta_cache[image_id]
        label = self._label_cache[image_id]

        return image, metadata, label

    def get_labels(self) -> List[int]:
        """Return all labels for this split (used for sampling weights)."""
        return [self._label_cache[img_id] for img_id in self.image_ids]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Get image transforms for a given split.

    Training uses data augmentation. Validation and test use only
    resize, centre crop, and normalisation.

    Args:
        split: One of 'train', 'val', or 'test'.
        image_size: Target image size.

    Returns:
        A torchvision Compose transform.
    """
    # ImageNet normalisation (for pretrained backbones)
    normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),  # slight oversize
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalise,
        ])


def create_dataloaders(
    metadata_path: str,
    image_dirs: List[str],
    splits: Dict[str, List[str]],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, MetadataEncoder]:
    """Create train/val/test DataLoaders with class-balanced sampling.

    Args:
        metadata_path: Path to HAM10000_metadata.csv.
        image_dirs: List of directories containing images.
        splits: Dictionary with 'train', 'val', 'test' image_id lists.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        image_size: Target image size.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, encoder).
    """
    df = pd.read_csv(metadata_path)

    # Fit encoder on training data only
    train_df = df[df["image_id"].isin(splits["train"])]
    encoder = MetadataEncoder().fit(train_df)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        dataset = HAM10000Dataset(
            image_ids=splits[split_name],
            metadata_df=df,
            image_dirs=image_dirs,
            encoder=encoder,
            transform=get_transforms(split_name, image_size),
        )

        if split_name == "train":
            # Weighted sampling: inverse class frequency
            labels = dataset.get_labels()
            class_counts = np.bincount(labels, minlength=len(DIAGNOSIS_LABELS))
            class_weights = 1.0 / (class_counts + 1e-6)
            sample_weights = [class_weights[l] for l in labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

    return loaders["train"], loaders["val"], loaders["test"], encoder
