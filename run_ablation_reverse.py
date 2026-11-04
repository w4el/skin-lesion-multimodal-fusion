"""Reverse attention direction ablation.

The methodology argues that image patches should be queries and metadata
tokens should be keys and values, on the grounds that this lets each
spatial region be contextualised by clinical information. That argument is
currently made on theoretical grounds only. This script tests it.

The reversed variant uses metadata tokens as queries and image patches as
keys and values, which reduces the attended sequence from 49 tokens to 3
before pooling. Everything else (backbone, d_model, heads, layers,
optimiser, schedule, loss, sampler, seeds, split) is held fixed.

Outputs use the model name 'cross_attention_reverse' so nothing written by
the original experiments is touched:
    results/cross_attention_reverse_seed{S}_best.pth
    results/cross_attention_reverse_seed{S}_test_results.json
    results/probs/cross_attention_reverse_seed{S}_probs.npy

Usage (PowerShell):
    python run_ablation_reverse.py
    python run_ablation_reverse.py -Seeds 42          # single seed smoke test
"""

import argparse
import json

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.analysis.common import (
    CLASS_NAMES,
    MEL_IDX,
    PROBS_DIR,
    RESULTS_DIR,
    SEEDS,
    ensure_dirs,
    evaluate_fp32,
    get_device,
    get_test_loader,
)
from src.models.architectures import build_model
from src.training.trainer import Trainer, get_class_weights
from src.utils.seed import set_seed

MODEL_NAME = "cross_attention_reverse"


def evaluate_and_save(model, test_loader, device, seed):
    """Full-precision test evaluation, saved in the same JSON schema as the
    original runs so that every downstream script works unchanged."""
    preds, labels, probs = evaluate_fp32(model, test_loader, device)

    metrics = {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auroc": float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        ),
    }
    mel_mask = labels == MEL_IDX
    metrics["mel_sensitivity"] = float(
        (preds[mel_mask] == MEL_IDX).sum() / mel_mask.sum()
    )

    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(labels, preds)

    results = {
        "metrics": metrics,
        "per_class_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
    }

    out_path = RESULTS_DIR / f"{MODEL_NAME}_seed{seed}_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    np.save(PROBS_DIR / f"{MODEL_NAME}_seed{seed}_probs.npy", probs)

    print(f"  Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"  AUROC:             {metrics['auroc']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Mel Sensitivity:   {metrics['mel_sensitivity']:.4f}")
    print(f"  Saved {out_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    ensure_dirs()
    device = get_device()

    train_loader, val_loader, test_loader, encoder = get_test_loader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    metadata_dim = encoder.metadata_dim
    print(f"Metadata dimension: {metadata_dim}")

    all_metrics = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Reverse-direction cross-attention, seed {seed}")
        print(f"{'='*60}")

        set_seed(seed)

        model = build_model(
            MODEL_NAME, metadata_dim=metadata_dim, num_classes=7
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.lr,
            num_epochs=args.num_epochs,
            patience=args.patience,
            class_weights=get_class_weights(train_loader),
            save_dir=str(RESULTS_DIR),
            experiment_name=f"{MODEL_NAME}_seed{seed}",
        )
        trainer.train()

        all_metrics.append(evaluate_and_save(model, test_loader, device, seed))

    # Aggregate
    summary = {"model_name": MODEL_NAME, "seeds": args.seeds,
               "per_seed_metrics": all_metrics, "aggregated": {}}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        summary["aggregated"][key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }

    with open(RESULTS_DIR / f"{MODEL_NAME}_multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Reverse-direction ablation summary")
    for key, vals in summary["aggregated"].items():
        print(f"  {key}: {vals['mean']:.4f} +/- {vals['std']:.4f}")
    print("Compare against the primary cross-attention row in the main results table.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()