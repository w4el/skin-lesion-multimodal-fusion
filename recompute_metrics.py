"""Recompute AUROC and full metrics from saved model checkpoints.

Since AUROC computation was silently failing during training runs,
this script reloads each saved best checkpoint, runs test-set evaluation
outside the autocast context, and saves complete metrics.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.data.dataset import create_dataloaders
from src.data.metadata import DIAGNOSIS_LABELS, DIAGNOSIS_NAMES
from src.data.split import load_splits
from src.models.architectures import build_model


def evaluate_checkpoint(model, test_loader, device):
    """Full evaluation outside autocast to get proper AUROC."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, metadata, labels in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            logits = model(images, metadata)
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = {
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "auroc": roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        ),
    }

    mel_idx = DIAGNOSIS_LABELS["mel"]
    mel_mask = all_labels == mel_idx
    metrics["mel_sensitivity"] = float(
        (all_preds[mel_mask] == mel_idx).sum() / mel_mask.sum()
    )

    return metrics, all_preds, all_labels, all_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    splits = load_splits("results/splits.json")
    _, _, test_loader, encoder = create_dataloaders(
        metadata_path="data/HAM10000_metadata.csv",
        image_dirs=["data/HAM10000_images_part_1", "data/HAM10000_images_part_2"],
        splits=splits,
        batch_size=32,
        num_workers=0,
    )

    metadata_dim = encoder.metadata_dim

    # Find all checkpoints
    results_dir = Path("results")
    checkpoints = sorted(results_dir.glob("*_seed*_best.pth"))
    print(f"Found {len(checkpoints)} checkpoints")

    all_results = {}

    for ckpt_path in checkpoints:
        # Parse model name and seed from filename
        stem = ckpt_path.stem  # e.g. "image_only_seed42_best"
        parts = stem.replace("_best", "").rsplit("_seed", 1)
        model_name = parts[0]
        seed = int(parts[1])
        exp_name = f"{model_name}_seed{seed}"

        print(f"\n{'='*60}")
        print(f"Evaluating {exp_name}")
        print(f"{'='*60}")

        # Build fresh model, load weights
        model = build_model(
            model_name, metadata_dim=metadata_dim, num_classes=7
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Evaluate
        metrics, preds, labels, probs = evaluate_checkpoint(
            model, test_loader, device
        )

        # Per-class report
        class_names = [DIAGNOSIS_NAMES[i] for i in range(7)]
        report = classification_report(
            labels, preds, target_names=class_names,
            output_dict=True, zero_division=0
        )
        conf_mat = confusion_matrix(labels, preds)

        # Print
        print(f"  Macro-F1:          {metrics['macro_f1']:.4f}")
        print(f"  AUROC:             {metrics['auroc']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Mel Sensitivity:   {metrics['mel_sensitivity']:.4f}")

        # Save updated results file
        results = {
            "metrics": metrics,
            "per_class_report": report,
            "confusion_matrix": conf_mat.tolist(),
            "predictions": preds.tolist(),
            "labels": labels.tolist(),
        }
        out_path = results_dir / f"{exp_name}_test_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Track for summary
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(metrics)

    # Print summary and update multi-seed summaries
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY (recomputed)")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Macro-F1':>16} {'AUROC':>16} {'Bal.Acc':>16} {'Mel.Sens':>16}")
    print("-" * 100)

    for model_name, metrics_list in all_results.items():
        # Aggregate
        keys = metrics_list[0].keys()
        aggregated = {}
        for k in keys:
            vals = [m[k] for m in metrics_list]
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }

        # Save multi-seed summary
        summary = {
            "model_name": model_name,
            "per_seed_metrics": metrics_list,
            "aggregated": aggregated,
        }
        summary_path = results_dir / f"{model_name}_multi_seed_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(
            f"{model_name:<20} "
            f"{aggregated['macro_f1']['mean']:.4f}+/-{aggregated['macro_f1']['std']:.4f} "
            f"{aggregated['auroc']['mean']:.4f}+/-{aggregated['auroc']['std']:.4f} "
            f"{aggregated['balanced_accuracy']['mean']:.4f}+/-{aggregated['balanced_accuracy']['std']:.4f} "
            f"{aggregated['mel_sensitivity']['mean']:.4f}+/-{aggregated['mel_sensitivity']['std']:.4f}"
        )


if __name__ == "__main__":
    main()