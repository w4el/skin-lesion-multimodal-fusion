"""Grad-CAM analysis for skin lesion classification models.

Generates Grad-CAM heatmaps for the image-only baseline and cross-attention
model on representative examples from the test set. Framed as plausibility
inspection: does the model attend to lesion regions or to artefacts like
rulers, hair, ink markings and vignetting?
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.data.dataset import create_dataloaders
from src.data.metadata import DIAGNOSIS_LABELS, DIAGNOSIS_NAMES
from src.data.split import load_splits
from src.models.architectures import build_model


def get_target_layer(model, model_name: str):
    """Get the target conv layer for Grad-CAM."""
    if model_name == "image_only":
        # EfficientNet-B0's last conv block
        return model.backbone.backbone.conv_head
    elif model_name == "cross_attention":
        # Same backbone, before patches are extracted
        return model.image_backbone.backbone.conv_head
    else:
        raise ValueError(f"Grad-CAM not supported for {model_name}")


class ModelWrapper(nn.Module):
    """Wrapper so Grad-CAM sees a single-input model.

    Grad-CAM expects model(image) -> logits, but our models take
    model(image, metadata). This wrapper fixes metadata to a specific
    value.
    """

    def __init__(self, model, metadata):
        super().__init__()
        self.model = model
        self.metadata = metadata

    def forward(self, image):
        # metadata may be a single row; broadcast to match batch
        batch_size = image.shape[0]
        if self.metadata.shape[0] != batch_size:
            metadata = self.metadata.expand(batch_size, -1)
        else:
            metadata = self.metadata
        return self.model(image, metadata)


def load_image_for_gradcam(
    image_id: str, image_dirs: List[str]
) -> Tuple[np.ndarray, torch.Tensor]:
    """Load and preprocess an image for Grad-CAM.

    Returns:
        rgb_image: Numpy array in [0, 1] for visualisation
        tensor_image: Normalised tensor for model input
    """
    # Find the image
    for d in image_dirs:
        path = Path(d) / f"{image_id}.jpg"
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"{image_id}.jpg not found")

    # Load
    pil_image = Image.open(path).convert("RGB")

    # For visualisation: resize and centre crop only
    viz_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    viz_image = np.array(viz_transform(pil_image)).astype(np.float32) / 255.0

    # For model input: full normalisation
    model_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensor_image = model_transform(pil_image).unsqueeze(0)

    return viz_image, tensor_image


def find_representative_samples(
    metadata_df: pd.DataFrame,
    splits: Dict,
    n_per_class: int = 2,
) -> Dict[str, List[str]]:
    """Find representative test samples per class."""
    test_ids = set(splits["test"])
    test_df = metadata_df[metadata_df["image_id"].isin(test_ids)]

    samples = {}
    for cls_name in DIAGNOSIS_LABELS.keys():
        cls_samples = test_df[test_df["dx"] == cls_name]["image_id"].tolist()
        # Take first n samples (deterministic)
        samples[cls_name] = cls_samples[:n_per_class]

    return samples


def generate_gradcam_grid(
    models: Dict[str, nn.Module],
    metadata_encoder,
    metadata_df: pd.DataFrame,
    samples: Dict[str, List[str]],
    image_dirs: List[str],
    device: torch.device,
    output_path: str,
):
    """Generate a grid of Grad-CAM visualisations.

    Rows = classes, Columns = original + heatmap per model.
    """
    n_classes = len(samples)
    n_models = len(models)
    n_cols = 1 + n_models  # original + one column per model

    # Take first sample per class for main grid
    fig, axes = plt.subplots(
        n_classes, n_cols, figsize=(4 * n_cols, 4 * n_classes)
    )
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    metadata_df_indexed = metadata_df.set_index("image_id")

    for row_idx, (cls_name, image_ids) in enumerate(samples.items()):
        if not image_ids:
            continue

        image_id = image_ids[0]
        viz_image, tensor_image = load_image_for_gradcam(image_id, image_dirs)
        tensor_image = tensor_image.to(device)

        # Get metadata for this image
        meta_row = metadata_df_indexed.loc[image_id]
        metadata = metadata_encoder.encode(meta_row).unsqueeze(0).to(device)

        true_label = DIAGNOSIS_LABELS[cls_name]

        # Original image
        axes[row_idx, 0].imshow(viz_image)
        axes[row_idx, 0].set_title(f"{cls_name.upper()}\n({image_id})", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Grad-CAM for each model
        for col_idx, (model_name, model) in enumerate(models.items(), start=1):
            model.eval()

            # Get model prediction
            with torch.no_grad():
                logits = model(tensor_image, metadata)
                pred = logits.argmax(dim=1).item()
                pred_name = DIAGNOSIS_NAMES[pred]

            # Set up Grad-CAM
            target_layer = get_target_layer(model, model_name)
            wrapper = ModelWrapper(model, metadata)

            with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
                targets = [ClassifierOutputTarget(true_label)]
                grayscale_cam = cam(input_tensor=tensor_image, targets=targets)[0]

            # Overlay heatmap on image
            cam_image = show_cam_on_image(
                viz_image, grayscale_cam, use_rgb=True
            )

            axes[row_idx, col_idx].imshow(cam_image)
            correct = "✓" if pred == true_label else "✗"
            axes[row_idx, col_idx].set_title(
                f"{model_name.replace('_', ' ').title()}\nPred: {pred_name} {correct}",
                fontsize=10,
            )
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def analyse_failure_cases(
    models: Dict[str, nn.Module],
    metadata_encoder,
    metadata_df: pd.DataFrame,
    splits: Dict,
    image_dirs: List[str],
    device: torch.device,
    output_dir: str,
):
    """Find and visualise samples where cross-attention misclassifies melanoma
    (clinically important false negatives)."""
    test_ids = set(splits["test"])
    test_df = metadata_df[metadata_df["image_id"].isin(test_ids)]
    mel_df = test_df[test_df["dx"] == "mel"]

    # Load cross-attention predictions from saved results
    import json
    with open("results/cross_attention_seed42_test_results.json") as f:
        results = json.load(f)

    labels = np.array(results["labels"])
    preds = np.array(results["predictions"])
    mel_idx = DIAGNOSIS_LABELS["mel"]

    # Find false negatives for melanoma (mel was true, model predicted otherwise)
    fn_mask = (labels == mel_idx) & (preds != mel_idx)
    fn_positions = np.where(fn_mask)[0]

    # Get image IDs for these failures
    # We need the order of the test loader
    from src.data.dataset import HAM10000Dataset, get_transforms
    dataset = HAM10000Dataset(
        image_ids=splits["test"],
        metadata_df=metadata_df,
        image_dirs=image_dirs,
        encoder=metadata_encoder,
        transform=get_transforms("test"),
    )

    # The saved predictions follow the DataLoader order (shuffle=False for test)
    fn_image_ids = [dataset.image_ids[i] for i in fn_positions[:6]]

    if not fn_image_ids:
        print("No melanoma false negatives found")
        return

    print(f"Found {fn_mask.sum()} melanoma false negatives; visualising first 6")

    # Grid: original + gradcam for each of 6 failures
    fig, axes = plt.subplots(6, 2, figsize=(8, 24))
    metadata_df_indexed = metadata_df.set_index("image_id")

    for row_idx, image_id in enumerate(fn_image_ids):
        viz_image, tensor_image = load_image_for_gradcam(image_id, image_dirs)
        tensor_image = tensor_image.to(device)

        meta_row = metadata_df_indexed.loc[image_id]
        metadata = metadata_encoder.encode(meta_row).unsqueeze(0).to(device)

        # Original
        axes[row_idx, 0].imshow(viz_image)
        axes[row_idx, 0].set_title(f"True: mel ({image_id})", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Grad-CAM on the WRONG prediction (what the model thought it was)
        model = models["cross_attention"]
        model.eval()
        with torch.no_grad():
            logits = model(tensor_image, metadata)
            pred = logits.argmax(dim=1).item()

        target_layer = get_target_layer(model, "cross_attention")
        wrapper = ModelWrapper(model, metadata)
        with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=tensor_image, targets=targets)[0]

        cam_image = show_cam_on_image(viz_image, grayscale_cam, use_rgb=True)
        axes[row_idx, 1].imshow(cam_image)
        axes[row_idx, 1].set_title(
            f"Cross-Attn pred: {DIAGNOSIS_NAMES[pred]}", fontsize=10
        )
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gradcam_failures_melanoma.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/gradcam_failures_melanoma.png")


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
    metadata_df = pd.read_csv(metadata_path)

    # Get metadata encoder
    _, _, _, encoder = create_dataloaders(
        metadata_path=metadata_path,
        image_dirs=image_dirs,
        splits=splits,
        batch_size=32,
        num_workers=0,
    )

    # Load trained models (use seed 42 checkpoints)
    models = {}
    for model_name in ["image_only", "cross_attention"]:
        model = build_model(
            model_name, metadata_dim=encoder.metadata_dim, num_classes=7
        ).to(device)
        ckpt = f"results/{model_name}_seed42_best.pth"
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
        models[model_name] = model
        print(f"Loaded {model_name}")

    # Find representative samples
    samples = find_representative_samples(metadata_df, splits, n_per_class=2)
    for cls, ids in samples.items():
        print(f"  {cls}: {len(ids)} samples")

    # Generate main comparison grid
    print("\nGenerating Grad-CAM comparison grid...")
    generate_gradcam_grid(
        models=models,
        metadata_encoder=encoder,
        metadata_df=metadata_df,
        samples=samples,
        image_dirs=image_dirs,
        device=device,
        output_path=str(output_dir / "gradcam_comparison.png"),
    )

    # Failure mode analysis
    print("\nAnalysing melanoma failure cases...")
    analyse_failure_cases(
        models=models,
        metadata_encoder=encoder,
        metadata_df=metadata_df,
        splits=splits,
        image_dirs=image_dirs,
        device=device,
        output_dir=str(output_dir),
    )

    print("\nDone. Outputs in results/xai/")


if __name__ == "__main__":
    main()
