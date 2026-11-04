"""Main experiment runner for the comparative study.

Runs all five model variants with multi-seed evaluation under the
same leakage-controlled protocol, then performs statistical comparisons.

Usage:
    python run_experiments.py --data_dir <path_to_ham10000> --model <model_name>

    Or run all models:
    python run_experiments.py --data_dir <path_to_ham10000> --model all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import create_dataloaders
from src.data.split import create_splits, load_splits
from src.training.statistics import compare_all_models
from src.training.trainer import run_multi_seed


def find_image_dirs(data_dir: str):
    """Find the image directories in the HAM10000 data folder."""
    data_path = Path(data_dir)
    # Common structures from Kaggle downloads
    possible_dirs = [
        # Flat structure
        [data_path],
        # Two-part structure
        [data_path / "HAM10000_images_part_1", data_path / "HAM10000_images_part_2"],
        # Nested structure
        [data_path / "ham10000_images_part_1", data_path / "ham10000_images_part_2"],
    ]

    for dirs in possible_dirs:
        if all(d.exists() for d in dirs):
            return [str(d) for d in dirs]

    # Fallback: search for directories containing .jpg files
    jpg_dirs = set()
    for jpg in data_path.rglob("*.jpg"):
        jpg_dirs.add(str(jpg.parent))

    if jpg_dirs:
        return list(jpg_dirs)

    raise FileNotFoundError(
        f"Could not find image directories in {data_dir}. "
        "Please check the dataset structure."
    )


def find_metadata_csv(data_dir: str) -> str:
    """Find the metadata CSV file."""
    data_path = Path(data_dir)
    candidates = list(data_path.rglob("HAM10000_metadata.csv")) + \
                 list(data_path.rglob("HAM10000_metadata"))

    if not candidates:
        # Try common alternative names
        candidates = list(data_path.rglob("*metadata*.csv"))

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        f"Could not find HAM10000_metadata.csv in {data_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run skin lesion experiments")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to HAM10000 dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["image_only", "metadata_only", "concat_fusion",
                 "gated_fusion", "cross_attention", "all"],
        help="Which model to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for multi-seed evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Max epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--focal_loss", action="store_true", help="Use focal loss"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers"
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Find data files
    metadata_path = find_metadata_csv(args.data_dir)
    image_dirs = ["data/HAM10000_images_part_1", "data/HAM10000_images_part_2"]
    print(f"Metadata: {metadata_path}")
    print(f"Image dirs: {image_dirs}")

    # Create or load splits
    split_path = Path(args.save_dir) / "splits.json"
    if split_path.exists():
        print(f"Loading existing splits from {split_path}")
        splits = load_splits(str(split_path))
    else:
        print("Creating new leakage-controlled splits...")
        splits = create_splits(
            metadata_path, str(split_path), seed=42
        )

    # Create data loaders
    train_loader, val_loader, test_loader, encoder = create_dataloaders(
        metadata_path=metadata_path,
        image_dirs=image_dirs,
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    metadata_dim = encoder.metadata_dim
    print(f"Metadata dimension: {metadata_dim}")

    # Define models to run
    if args.model == "all":
        models_to_run = [
            "image_only", "metadata_only", "concat_fusion",
            "gated_fusion", "cross_attention",
        ]
    else:
        models_to_run = [args.model]

    # Run experiments
    all_summaries = {}
    for model_name in models_to_run:
        model_kwargs = {"metadata_dim": metadata_dim, "num_classes": 7}

        summary = run_multi_seed(
            model_name=model_name,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            seeds=args.seeds,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            patience=args.patience,
            use_focal_loss=args.focal_loss,
            save_dir=args.save_dir,
        )
        all_summaries[model_name] = summary

    # Statistical comparisons (if running all models)
    if len(models_to_run) > 1 and "cross_attention" in models_to_run:
        print("\n" + "=" * 60)
        print("Statistical Comparisons (cross-attention vs baselines)")
        print("=" * 60)

        # Load predictions from the first seed for comparison
        predictions = {}
        for model_name in models_to_run:
            result_path = (
                Path(args.save_dir) /
                f"{model_name}_seed{args.seeds[0]}_test_results.json"
            )
            with open(result_path) as f:
                data = json.load(f)
            predictions[model_name] = np.array(data["predictions"])
            test_labels = np.array(data["labels"])

        comparisons = compare_all_models(
            predictions, test_labels, reference_model="cross_attention"
        )

        for comp_name, comp_results in comparisons.items():
            print(f"\n{comp_name}:")
            mc = comp_results["mcnemar"]
            print(f"  McNemar: statistic={mc['statistic']:.3f}, p={mc['p_value']:.4f}")
            bs = comp_results["bootstrap_f1"]
            print(
                f"  Bootstrap F1 diff: {bs['observed_diff']:.4f} "
                f"[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}] "
                f"p={bs['p_value']:.4f}"
            )

        # Save comparisons
        comp_path = Path(args.save_dir) / "statistical_comparisons.json"
        with open(comp_path, "w") as f:
            json.dump(comparisons, f, indent=2, default=str)

    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Macro-F1':>12} {'AUROC':>12} {'Bal.Acc':>12} {'Mel.Sens':>12}")
    print("-" * 80)
    for model_name, summary in all_summaries.items():
        agg = summary["aggregated"]
        print(
            f"{model_name:<20} "
            f"{agg['macro_f1']['mean']:.4f}+/-{agg['macro_f1']['std']:.4f} "
            f"{agg['auroc']['mean']:.4f}+/-{agg['auroc']['std']:.4f} "
            f"{agg['balanced_accuracy']['mean']:.4f}+/-{agg['balanced_accuracy']['std']:.4f} "
            f"{agg['mel_sensitivity']['mean']:.4f}+/-{agg['mel_sensitivity']['std']:.4f}"
        )


if __name__ == "__main__":
    main()