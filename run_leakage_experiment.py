"""Quantify the effect of lesion-level leakage directly.

The paper argues that evaluation protocol, rather than architecture, drives
much of the variation in reported HAM10000 results, and offers two
explanations for why cross-attention's reported advantage over
concatenation disappears under a grouped split:

  (1) prior gains were partly leakage, exploited more by the more
      expressive model, so removing leakage compresses the gap;
  (2) cross-attention's advantage is real but confined to ranking quality.

As written, the paper cannot separate these. This script separates them, by
retraining concatenation and cross-attention on a deliberately non-grouped
split and measuring whether the gap between them widens.

If the gap widens under leakage, explanation (1) is supported and the paper
has directly measured the size of the effect it is arguing about. If the
gap does not widen, explanation (1) is weakened and (2) carries the
argument. Either outcome is reportable and both are stronger than the
current hedge.

The leaky split matches the grouped split exactly in partition sizes and
class stratification. The only difference is that images are assigned
independently rather than by lesion, so images of the same lesion can
appear in both training and test. That single controlled difference is the
point.

Outputs, all under names that cannot collide with the primary runs:
    results/splits_leaky.json
    results/leakage_overlap_stats.json
    results/{concat_fusion,cross_attention}_leaky_seed{S}_best.pth
    results/{concat_fusion,cross_attention}_leaky_seed{S}_test_results.json
    results/probs/..._leaky_seed{S}_probs.npy
    results/{model}_leaky_multi_seed_summary.json

Usage (PowerShell), from the project root:
    python -u run_leakage_experiment.py --split_only     # build and inspect first
    python -u run_leakage_experiment.py                  # full run
    python -u run_leakage_experiment.py --seeds 42       # single seed, faster
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.analysis.common import (
    CLASS_NAMES,
    IMAGE_DIRS,
    MEL_IDX,
    METADATA_PATH,
    PROBS_DIR,
    RESULTS_DIR,
    SEEDS,
    SPLITS_LEAKY_PATH,
    SPLITS_PATH,
    ensure_dirs,
    evaluate_fp32,
    get_device,
)
from src.data.dataset import create_dataloaders
from src.data.split import load_splits
from src.models.architectures import build_model
from src.training.trainer import Trainer, get_class_weights
from src.utils.seed import set_seed

SPLIT_SEED = 42
MODELS = ["concat_fusion", "cross_attention"]


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def _extract_ids(part):
    """Pull a list of image ids out of whatever shape the split file uses."""
    if isinstance(part, list):
        return list(part)
    if isinstance(part, dict):
        for key in ("image_id", "image_ids", "ids", "images"):
            if key in part:
                return list(part[key])
    raise ValueError(
        f"Could not read image ids from split partition of type {type(part)}. "
        f"Inspect results/splits.json and tell me its structure."
    )


def _rebuild(part_template, ids):
    """Write ids back in the same shape the original split file used."""
    if isinstance(part_template, list):
        return list(ids)
    out = dict(part_template)
    for key in ("image_id", "image_ids", "ids", "images"):
        if key in out:
            out[key] = list(ids)
            return out
    raise ValueError("Unexpected split partition structure.")


def build_leaky_split(verbose=True):
    """Create a random, non-grouped stratified split with the same sizes."""
    grouped = load_splits(SPLITS_PATH)
    if not all(k in grouped for k in ("train", "val", "test")):
        raise ValueError(f"Unexpected split keys: {list(grouped)}")

    sizes = {k: len(_extract_ids(grouped[k])) for k in ("train", "val", "test")}
    all_ids = []
    for k in ("train", "val", "test"):
        all_ids.extend(_extract_ids(grouped[k]))

    meta = pd.read_csv(METADATA_PATH)
    id_col = "image_id" if "image_id" in meta.columns else meta.columns[1]
    meta = meta.set_index(id_col)

    missing = [i for i in all_ids if i not in meta.index]
    if missing:
        raise ValueError(f"{len(missing)} split ids not found in metadata, e.g. {missing[:3]}")

    df = meta.loc[all_ids]
    labels = df["dx"].values
    lesions = df["lesion_id"].values

    n_total = len(all_ids)
    n_test = sizes["test"]
    n_val = sizes["val"]

    # Two stratified splits, ignoring lesion_id entirely. This is exactly the
    # procedure the leakage audit identifies in prior work.
    idx = np.arange(n_total)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=n_test, stratify=labels, random_state=SPLIT_SEED
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=n_val,
        stratify=labels[idx_trainval], random_state=SPLIT_SEED
    )

    ids = np.array(all_ids)
    leaky = {
        "train": _rebuild(grouped["train"], ids[idx_train].tolist()),
        "val": _rebuild(grouped["val"], ids[idx_val].tolist()),
        "test": _rebuild(grouped["test"], ids[idx_test].tolist()),
    }
    for k, v in grouped.items():
        leaky.setdefault(k, v)

    Path(SPLITS_LEAKY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_LEAKY_PATH, "w") as f:
        json.dump(leaky, f, indent=2)

    stats = _overlap_stats(lesions, labels, idx_train, idx_val, idx_test, sizes)
    with open(RESULTS_DIR / "leakage_overlap_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        _report(stats, sizes)

    return leaky, stats


def _overlap_stats(lesions, labels, idx_train, idx_val, idx_test, sizes):
    train_lesions = set(lesions[idx_train])
    test_lesions = set(lesions[idx_test])
    val_lesions = set(lesions[idx_val])

    contaminated_test = int(np.sum([lesions[i] in train_lesions for i in idx_test]))
    contaminated_val = int(np.sum([lesions[i] in train_lesions for i in idx_val]))

    # Class balance check: the leaky split must match the grouped one, or the
    # comparison is confounded by something other than leakage.
    def dist(idxs):
        vals, counts = np.unique(labels[idxs], return_counts=True)
        return {str(v): round(float(c) / len(idxs), 4) for v, c in zip(vals, counts)}

    return {
        "sizes": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
        "target_sizes": sizes,
        "n_test_images_sharing_lesion_with_train": contaminated_test,
        "pct_test_contaminated": round(100 * contaminated_test / len(idx_test), 2),
        "n_val_images_sharing_lesion_with_train": contaminated_val,
        "pct_val_contaminated": round(100 * contaminated_val / len(idx_val), 2),
        "n_lesions_spanning_train_and_test": len(train_lesions & test_lesions),
        "n_lesions_spanning_train_and_val": len(train_lesions & val_lesions),
        "class_distribution_train": dist(idx_train),
        "class_distribution_test": dist(idx_test),
    }


def _report(stats, sizes):
    print("\n" + "=" * 72)
    print("LEAKY SPLIT CONSTRUCTED")
    print("=" * 72)
    print(f"  Partition sizes: {stats['sizes']}  (target {stats['target_sizes']})")
    print(f"  Test images sharing a lesion with training: "
          f"{stats['n_test_images_sharing_lesion_with_train']} "
          f"({stats['pct_test_contaminated']}%)")
    print(f"  Lesions spanning train and test: "
          f"{stats['n_lesions_spanning_train_and_test']}")
    print(f"  Val images sharing a lesion with training: "
          f"{stats['n_val_images_sharing_lesion_with_train']} "
          f"({stats['pct_val_contaminated']}%)")
    print("\n  Class distribution, train:", stats["class_distribution_train"])
    print("  Class distribution, test: ", stats["class_distribution_test"])
    print("\n  These two distributions should match closely. If they do not,")
    print("  stop: the comparison would be confounded by class balance rather")
    print("  than by leakage alone.")
    if stats["n_test_images_sharing_lesion_with_train"] == 0:
        print("\n  WARNING: no contamination produced. Something is wrong;")
        print("  the whole experiment depends on this being non-zero.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
        "split": "leaky_non_grouped",
    }
    out = RESULTS_DIR / f"{name}_seed{seed}_test_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    np.save(PROBS_DIR / f"{name}_seed{seed}_probs.npy", probs)

    print(f"  Macro-F1: {metrics['macro_f1']:.4f} | AUROC: {metrics['auroc']:.4f} "
          f"| Bal.Acc: {metrics['balanced_accuracy']:.4f} "
          f"| Mel.Sens: {metrics['mel_sensitivity']:.4f}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_only", action="store_true",
                   help="Build and inspect the leaky split without training")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    ensure_dirs()
    leaky, stats = build_leaky_split()

    if args.split_only:
        print("\nSplit built. Check the numbers above, then run without "
              "--split_only to train.")
        return

    device = get_device()
    train_loader, val_loader, test_loader, encoder = create_dataloaders(
        metadata_path=METADATA_PATH,
        image_dirs=IMAGE_DIRS,
        splits=leaky,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metadata_dim = encoder.metadata_dim
    class_weights = get_class_weights(train_loader)
    print(f"Metadata dimension: {metadata_dim}")

    summary = {}
    for base in args.models:
        name = f"{base}_leaky"
        per_seed = []
        for seed in args.seeds:
            print(f"\n{'='*72}\n{name}, seed {seed}\n{'='*72}")
            set_seed(seed)
            model = build_model(base, metadata_dim=metadata_dim, num_classes=7)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                learning_rate=args.lr,
                num_epochs=args.num_epochs,
                patience=args.patience,
                class_weights=class_weights,
                save_dir=str(RESULTS_DIR),
                experiment_name=f"{name}_seed{seed}",
            )
            trainer.train()
            per_seed.append(evaluate_and_save(model, test_loader, device, name, seed))

        agg = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
                   "std": float(np.std([m[k] for m in per_seed])),
                   "values": [m[k] for m in per_seed]}
               for k in per_seed[0]}
        summary[name] = agg
        with open(RESULTS_DIR / f"{name}_multi_seed_summary.json", "w") as f:
            json.dump({"model_name": name, "seeds": args.seeds,
                       "per_seed_metrics": per_seed, "aggregated": agg,
                       "leakage_stats": stats}, f, indent=2, default=str)

    print(f"\n{'='*72}\nLEAKAGE EXPERIMENT SUMMARY\n{'='*72}")
    for name, agg in summary.items():
        print(f"\n{name}")
        for k, v in agg.items():
            print(f"  {k:<20} {v['mean']:.4f} +/- {v['std']:.4f}")

    if len(summary) == 2:
        c = summary.get("concat_fusion_leaky")
        x = summary.get("cross_attention_leaky")
        if c and x:
            print(f"\n{'-'*72}")
            print("THE NUMBER THAT MATTERS")
            print(f"{'-'*72}")
            for k in ("macro_f1", "balanced_accuracy", "auroc"):
                gap = x[k]["mean"] - c[k]["mean"]
                print(f"  {k}: cross-attention minus concatenation = {gap:+.4f}  (leaky split)")
            print("\n  Compare each against the same gap on the grouped split,")
            print("  under the grouped split. If the gap is larger here, leakage was")
            print("  inflating the advantage of the more expressive model.")
            print(f"{'-'*72}")


if __name__ == "__main__":
    main()