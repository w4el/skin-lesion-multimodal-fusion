"""Generate the result figures that the paper is currently missing.

    figures/confusion_matrices.png   row-normalised, averaged over seeds
    figures/roc_per_class.png        per-class ROC for cross-attention
    figures/roc_melanoma.png         melanoma ROC, all models overlaid
    figures/calibration.png          reliability diagrams

Confusion matrices come straight from the stored JSONs. The ROC and
reliability figures need results/probs/, so run run_export_probabilities.py
first.

ROC and reliability curves are drawn from a single seed (42 by default),
because averaging curves across seeds is not well defined; the
seed-to-seed spread is reported numerically in Tables I and V instead.

Usage (PowerShell):
    python run_make_figures.py
    python run_make_figures.py --seed 123
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from src.analysis.common import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    FIGURES_DIR,
    MEL_IDX,
    SEEDS,
    available_models,
    ensure_dirs,
    load_predictions,
    load_probs,
    load_results,
)

DPI = 200


def fig_confusion_matrices(models):
    """Row-normalised confusion matrices, averaged across seeds."""
    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, models):
        mats = []
        for seed in SEEDS:
            cm = np.array(load_results(m, seed)["confusion_matrix"], dtype=float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mats.append(cm / row_sums)
        cm = np.mean(mats, axis=0)

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(DISPLAY_NAMES[m], fontsize=11)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")

    for ax in axes[len(models):]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02,
                 label="Proportion of true class")
    out = FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_roc_per_class(model, seed):
    probs = load_probs(model, seed)
    _, labels = load_predictions(model, seed)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, name in enumerate(CLASS_NAMES):
        y_true = (labels == i).astype(int)
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        ax.plot(fpr, tpr, lw=1.6,
                label=f"{name} (AUC {auc(fpr, tpr):.3f}, n={int(y_true.sum())})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Per-class ROC, {DISPLAY_NAMES[model]} (seed {seed})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    out = FIGURES_DIR / "roc_per_class.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_roc_melanoma(models, seed):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for m in models:
        probs = load_probs(m, seed)
        _, labels = load_predictions(m, seed)
        y_true = (labels == MEL_IDX).astype(int)
        fpr, tpr, _ = roc_curve(y_true, probs[:, MEL_IDX])
        ax.plot(fpr, tpr, lw=1.6,
                label=f"{DISPLAY_NAMES[m]} (AUC {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Melanoma ROC across models (seed {seed})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    out = FIGURES_DIR / "roc_melanoma.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_calibration(models, seed, n_bins=10):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    bins = np.linspace(0, 1, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2

    for m in models:
        probs = load_probs(m, seed)
        _, labels = load_predictions(m, seed)
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(float)

        xs, ys = [], []
        for lo, hi, c in zip(bins[:-1], bins[1:], centres):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() < 10:
                continue
            xs.append(conf[mask].mean())
            ys.append(correct[mask].mean())
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, label=DISPLAY_NAMES[m])

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram (seed {seed}, bins with $n\\geq10$)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    out = FIGURES_DIR / "calibration.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    models = available_models()
    print(f"Models: {', '.join(models)}")

    print("\nConfusion matrices...")
    fig_confusion_matrices(models)

    try:
        print("\nROC and calibration...")
        fig_roc_per_class("cross_attention", args.seed)
        fig_roc_melanoma(models, args.seed)
        fig_calibration(models, args.seed)
    except FileNotFoundError as e:
        print(f"\n  Skipped: {e}")

    print("\nDone. Figures in figures/")
    print("Remember to copy the XAI figures across as well:")
    print("  Copy-Item results\\xai\\*.png figures\\")


if __name__ == "__main__":
    main()
