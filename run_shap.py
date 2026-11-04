"""SHAP analysis for metadata contributions in fusion models.

Uses KernelSHAP to quantify the contribution of each metadata feature
(age, sex, localisation) to the model's predictions in the cross-attention
fusion model. Framed as plausibility inspection: do metadata contributions
align with clinical priors?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from src.data.dataset import create_dataloaders, HAM10000Dataset, get_transforms
from src.data.metadata import (
    DIAGNOSIS_NAMES,
    LOCALIZATION_CATEGORIES,
    SEX_CATEGORIES,
)
from src.data.split import load_splits
from src.models.architectures import build_model
from src.xai.shap_utils import MetadataOnlyWrapper


# Feature names matching the encoder output order
FEATURE_NAMES = (
    ["age_normalised", "age_missing"]
    + [f"sex_{s}" for s in SEX_CATEGORIES]
    + [f"loc_{l.replace(' ', '_')}" for l in LOCALIZATION_CATEGORIES]
)


def compute_shap_values(
    model,
    dataset: HAM10000Dataset,
    device: torch.device,
    n_background: int = 50,
    n_explain: int = 100,
) -> np.ndarray:
    """Compute SHAP values across a sample of test images.

    For each test image, we compute SHAP values for the metadata features
    with the image fixed. We then aggregate across images.

    Args:
        model: Trained fusion model.
        dataset: Test dataset.
        device: Torch device.
        n_background: Number of background samples for KernelSHAP.
        n_explain: Number of test samples to explain.

    Returns:
        SHAP values of shape (n_explain, n_features, n_classes).
    """
    # Gather background metadata (representative sample)
    n_total = len(dataset)
    bg_indices = np.random.RandomState(42).choice(n_total, n_background, replace=False)
    background_metadata = np.stack([
        dataset._meta_cache[dataset.image_ids[i]].numpy() for i in bg_indices
    ])

    # Select images to explain (deterministic sample of test set)
    explain_indices = np.random.RandomState(0).choice(
        n_total, min(n_explain, n_total), replace=False
    )

    all_shap_values = []
    for i, idx in enumerate(explain_indices):
        if i % 10 == 0:
            print(f"  SHAP explanation {i+1}/{len(explain_indices)}")

        # Load image and metadata for this sample
        image, metadata, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)
        metadata_np = metadata.numpy().reshape(1, -1)

        # Build wrapper for this specific image
        wrapper = MetadataOnlyWrapper(model, image)

        # KernelSHAP
        explainer = shap.KernelExplainer(
            wrapper, background_metadata, silent=True
        )
        shap_values = explainer.shap_values(
            metadata_np, nsamples=100, silent=True
        )

        # shap_values is either a list of arrays (one per class) or a 3D array
        if isinstance(shap_values, list):
            # (n_classes, 1, n_features) -> (n_features, n_classes)
            arr = np.stack([sv[0] for sv in shap_values], axis=-1)
        else:
            # (1, n_features, n_classes)
            arr = shap_values[0]

        all_shap_values.append(arr)

    return np.stack(all_shap_values)  # (n_explain, n_features, n_classes)


def plot_class_feature_importance(
    shap_values: np.ndarray, output_path: str
):
    """Plot mean absolute SHAP value per feature per class."""
    # shap_values: (n_samples, n_features, n_classes)
    mean_abs = np.abs(shap_values).mean(axis=0)  # (n_features, n_classes)

    fig, ax = plt.subplots(figsize=(12, 8))
    class_names = [DIAGNOSIS_NAMES[i] for i in range(7)]
    im = ax.imshow(mean_abs, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES)
    ax.set_xlabel("Class")
    ax.set_ylabel("Metadata Feature")
    ax.set_title("Mean |SHAP value| per feature per class (Cross-Attention)")

    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_top_features_per_class(
    shap_values: np.ndarray, output_path: str, top_k: int = 8
):
    """Bar chart of top-K features by mean |SHAP| for each class."""
    mean_abs = np.abs(shap_values).mean(axis=0)  # (n_features, n_classes)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for cls_idx in range(7):
        ax = axes[cls_idx]
        importances = mean_abs[:, cls_idx]
        top_idx = np.argsort(importances)[-top_k:][::-1]

        top_names = [FEATURE_NAMES[i] for i in top_idx]
        top_values = importances[top_idx]

        ax.barh(range(top_k), top_values[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Class: {DIAGNOSIS_NAMES[cls_idx]}")

    # Hide the 8th subplot
    axes[7].axis("off")

    plt.suptitle(
        "Top metadata features by SHAP importance per class (Cross-Attention)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path("results/xai")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    metadata_path = "data/HAM10000_metadata.csv"
    image_dirs = [
        "data/HAM10000_images_part_1",
        "data/HAM10000_images_part_2",
    ]
    splits = load_splits("results/splits.json")

    # Get encoder
    _, _, _, encoder = create_dataloaders(
        metadata_path=metadata_path,
        image_dirs=image_dirs,
        splits=splits,
        batch_size=32,
        num_workers=0,
    )

    # Load cross-attention model (seed 42)
    model = build_model(
        "cross_attention",
        metadata_dim=encoder.metadata_dim,
        num_classes=7,
    ).to(device)
    ckpt = "results/cross_attention_seed42_best.pth"
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    print("Loaded cross_attention model")

    # Build test dataset
    metadata_df = pd.read_csv(metadata_path)
    test_dataset = HAM10000Dataset(
        image_ids=splits["test"],
        metadata_df=metadata_df,
        image_dirs=image_dirs,
        encoder=encoder,
        transform=get_transforms("test"),
    )

    # Compute SHAP values
    print("\nComputing SHAP values (this takes a few minutes)...")
    shap_values = compute_shap_values(
        model=model,
        dataset=test_dataset,
        device=device,
        n_background=30,
        n_explain=50,
    )

    # Save raw values
    np.save(output_dir / "shap_values.npy", shap_values)
    print(f"Saved raw SHAP values: {shap_values.shape}")

    # Generate visualisations
    plot_class_feature_importance(
        shap_values, str(output_dir / "shap_feature_class_heatmap.png")
    )
    plot_top_features_per_class(
        shap_values, str(output_dir / "shap_top_features_per_class.png")
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SHAP SUMMARY: Top 3 features per class (mean |SHAP|)")
    print("=" * 60)
    mean_abs = np.abs(shap_values).mean(axis=0)
    for cls_idx in range(7):
        cls_name = DIAGNOSIS_NAMES[cls_idx]
        top_idx = np.argsort(mean_abs[:, cls_idx])[-3:][::-1]
        top = [f"{FEATURE_NAMES[i]} ({mean_abs[i, cls_idx]:.4f})" for i in top_idx]
        print(f"  {cls_name}: {', '.join(top)}")


if __name__ == "__main__":
    main()