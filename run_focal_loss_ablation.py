"""Focal loss ablation.

Swaps the class-weighted cross-entropy term for focal loss and leaves
everything else fixed.

What this ablation actually varies. The Trainer's focal loss path computes
cross-entropy without class weights, so switching it on does not add a
third imbalance correction on top of the existing two; it replaces the
class-weighted term. The comparison is therefore:

    primary   : class-weighted cross-entropy + weighted random sampler
    this run  : focal loss (unweighted) + weighted random sampler

That is a clean single-variable swap of the loss function, which is the
only interpretable version of this ablation. Stacking focal loss on top of
class weights would change two things at once.

Runs on cross-attention by default, since that is the model under study.
Other models can be added with --models.

Outputs, named so they cannot collide with the primary runs:
    results/cross_attention_focal_seed{S}_best.pth
    results/cross_attention_focal_seed{S}_test_results.json
    results/probs/cross_attention_focal_seed{S}_probs.npy
    results/cross_attention_focal_multi_seed_summary.json

Usage (PowerShell), from the project root:
    python -u run_focal_loss_ablation.py --check     # verify the Trainer API first
    python -u run_focal_loss_ablation.py
    python -u run_focal_loss_ablation.py --models cross_attention concat_fusion
"""

import argparse
import inspect
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
    DISPLAY_NAMES,
    MEL_IDX,
    PROBS_DIR,
    RESULTS_DIR,
    SEEDS,
    ensure_dirs,
    evaluate_fp32,
    get_device,
    get_test_loader,
)
from src.models import architectures
from src.models.architectures import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed

# Candidate names for the focal-loss switch on Trainer.__init__, in order of
# likelihood. Detected at runtime rather than assumed.
FOCAL_KWARG_CANDIDATES = ["use_focal_loss", "focal_loss", "use_focal", "focal"]
GAMMA_KWARG_CANDIDATES = ["focal_gamma", "gamma"]

DEFAULT_MODELS = ["cross_attention"]


def register_aliases(models):
    """Give each focal run its own registry name so nothing is overwritten."""
    for base in models:
        alias = f"{base}_focal"
        if base in architectures.MODEL_REGISTRY:
            architectures.MODEL_REGISTRY[alias] = architectures.MODEL_REGISTRY[base]
            DISPLAY_NAMES[alias] = DISPLAY_NAMES.get(base, base) + " (focal loss)"


def detect_kwargs():
    """Find what the Trainer actually calls its focal-loss switch."""
    params = inspect.signature(Trainer.__init__).parameters
    focal = next((k for k in FOCAL_KWARG_CANDIDATES if k in params), None)
    gamma = next((k for k in GAMMA_KWARG_CANDIDATES if k in params), None)

    if focal is None:
        raise SystemExit(
            "Could not find a focal-loss argument on Trainer.__init__.\n"
            f"  Tried: {FOCAL_KWARG_CANDIDATES}\n"
            f"  Trainer accepts: {[p for p in params if p != 'self']}\n"
            "Send me that list and I will correct the script."
        )
    return focal, gamma


def check():
    """Fast preflight. Confirms the switch exists before spending GPU hours."""
    focal, gamma = detect_kwargs()
    print(f"Focal-loss argument detected: {focal}")
    print(f"Gamma argument detected:      {gamma or 'none (Trainer default will be used)'}")

    has_method = any(
        n in dir(Trainer) for n in ("_focal_loss", "focal_loss", "_compute_focal_loss")
    )
    print(f"Focal loss method present on Trainer: {has_method}")
    if not has_method:
        print("  WARNING: no focal loss method found by name. The switch may")
        print("  still work; check the Trainer source before the full run.")

    params = inspect.signature(Trainer.__init__).parameters
    cw = params.get("class_weights")
    print(f"class_weights parameter: "
          f"{'present, default ' + repr(cw.default) if cw else 'ABSENT'}")
    print("\nPreflight passed. Run without --check to train.")


def evaluate_and_save(model, test_loader, device, name, seed):
    preds, labels, probs = evaluate_fp32(model, test_loader, device)
    metrics = {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
    }
    mel = labels == MEL_IDX
    metrics["mel_sensitivity"] = float((preds[mel] == MEL_IDX).sum() / max(1, mel.sum()))

    results = {
        "metrics": metrics,
        "per_class_report": classification_report(
            labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
        "loss": "focal_unweighted",
    }
    with open(RESULTS_DIR / f"{name}_seed{seed}_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    np.save(PROBS_DIR / f"{name}_seed{seed}_probs.npy", probs)

    print(f"  Macro-F1: {metrics['macro_f1']:.4f} | AUROC: {metrics['auroc']:.4f} "
          f"| Bal.Acc: {metrics['balanced_accuracy']:.4f} "
          f"| Mel.Sens: {metrics['mel_sensitivity']:.4f}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Preflight only: confirm the Trainer exposes focal loss")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    if args.check:
        check()
        return

    focal_kw, gamma_kw = detect_kwargs()
    ensure_dirs()
    register_aliases(args.models)
    device = get_device()

    train_loader, val_loader, test_loader, encoder = get_test_loader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    metadata_dim = encoder.metadata_dim
    print(f"Metadata dimension: {metadata_dim}")
    print(f"Loss: focal (gamma={args.gamma}), class weights disabled, "
          f"weighted sampler retained")

    for base in args.models:
        name = f"{base}_focal"
        per_seed = []

        for seed in args.seeds:
            print(f"\n{'='*72}\n{name}, seed {seed}\n{'='*72}")
            set_seed(seed)

            model = build_model(base, metadata_dim=metadata_dim, num_classes=7)

            kwargs = dict(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                learning_rate=args.lr,
                num_epochs=args.num_epochs,
                patience=args.patience,
                class_weights=None,   # focal loss replaces the weighted term
                save_dir=str(RESULTS_DIR),
                experiment_name=f"{name}_seed{seed}",
            )
            kwargs[focal_kw] = True
            if gamma_kw:
                kwargs[gamma_kw] = args.gamma

            trainer = Trainer(**kwargs)
            trainer.train()
            per_seed.append(evaluate_and_save(model, test_loader, device, name, seed))

        agg = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
                   "std": float(np.std([m[k] for m in per_seed])),
                   "values": [m[k] for m in per_seed]}
               for k in per_seed[0]}

        with open(RESULTS_DIR / f"{name}_multi_seed_summary.json", "w") as f:
            json.dump({"model_name": name, "seeds": args.seeds,
                       "loss": "focal_unweighted", "gamma": args.gamma,
                       "per_seed_metrics": per_seed, "aggregated": agg},
                      f, indent=2, default=str)

        print(f"\n{'='*72}\n{name} summary\n{'='*72}")
        for k, v in agg.items():
            print(f"  {k:<20} {v['mean']:.4f} +/- {v['std']:.4f}")
        print("\n  Compare against the same model's row in the main results table,")
        print("  which used class-weighted cross-entropy with the same sampler.")


if __name__ == "__main__":
    main()