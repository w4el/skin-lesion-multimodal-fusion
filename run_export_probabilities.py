"""Export full-precision test-set probabilities for every saved checkpoint.

recompute_metrics.py recomputes metrics but discards the probability
matrix, and evaluate_on_test() strips probabilities before writing JSON.
Per-class ROC curves and calibration analysis both need them, so this
script runs inference once per checkpoint and caches the result.

This is inference only. On 1,494 test images across 15 to 18 checkpoints it
takes a few minutes on the 4060 Ti. Nothing in results/ is overwritten;
output goes to results/probs/.

Usage (PowerShell):
    python run_export_probabilities.py
    python run_export_probabilities.py --force     # recompute existing
"""

import argparse

import numpy as np

from src.analysis.common import (
    PROBS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    evaluate_fp32,
    get_device,
    get_test_loader,
    load_predictions,
    load_checkpoint_model,
    probs_path,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include_leaky", action="store_true",
        help="Also export for models trained on the leaky split. Off by "
             "default: those models were evaluated on the non-grouped test "
             "set, so running them through the grouped loader would produce "
             "probabilities that do not correspond to their stored results.")
    args = parser.parse_args()

    ensure_dirs()
    device = get_device()

    _, _, test_loader, encoder = get_test_loader(num_workers=0)
    metadata_dim = encoder.metadata_dim

    checkpoints = sorted(RESULTS_DIR.glob("*_seed*_best.pth"))
    print(f"Found {len(checkpoints)} checkpoints")

    for ckpt in checkpoints:
        stem = ckpt.stem.replace("_best", "")
        model_name, seed_str = stem.rsplit("_seed", 1)
        seed = int(seed_str)

        if "_leaky" in model_name and not args.include_leaky:
            print(f"  Skipping {model_name} seed {seed} "
                  f"(leaky split, use --include_leaky to override)")
            continue

        out = probs_path(model_name, seed)
        if out.exists() and not args.force:
            print(f"  {out.name} exists, skipping")
            continue

        print(f"\nExporting {model_name} seed {seed}")
        model = load_checkpoint_model(model_name, seed, metadata_dim, device)
        preds, labels, probs = evaluate_fp32(model, test_loader, device)

        # Consistency check against the stored predictions. If this fails,
        # the saved JSON and the checkpoint have drifted apart and the
        # tables cannot be trusted.
        try:
            stored_preds, stored_labels = load_predictions(model_name, seed)
            if not np.array_equal(labels, stored_labels):
                print("  WARNING: label order differs from stored results")
            agreement = float((preds == stored_preds).mean())
            if agreement < 1.0:
                print(
                    f"  WARNING: predictions agree with stored JSON on only "
                    f"{agreement:.4%} of samples"
                )
            else:
                print("  Predictions match stored results exactly")
        except FileNotFoundError:
            print("  No stored results JSON to cross-check against")

        np.save(out, probs)
        print(f"  Saved {out}  shape={probs.shape}")

    print(f"\nDone. Probabilities in {PROBS_DIR}/")


if __name__ == "__main__":
    main()
