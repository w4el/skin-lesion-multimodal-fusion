"""Run statistical comparisons across all models.

Uses McNemar's test and bootstrapped 95% CIs on Macro-F1 differences.
Runs pairwise comparisons for the primary seed (42), and also aggregated
comparisons pooling predictions across seeds.
"""

import json
from pathlib import Path
from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from src.training.statistics import mcnemar_test, bootstrap_metric_diff


MODELS = ["image_only", "metadata_only", "concat_fusion", "gated_fusion", "cross_attention"]
SEEDS = [42, 123, 456]
RESULTS_DIR = Path("results")


def load_predictions(model_name: str, seed: int):
    """Load predictions and labels for a specific model + seed."""
    path = RESULTS_DIR / f"{model_name}_seed{seed}_test_results.json"
    with open(path) as f:
        data = json.load(f)
    return (
        np.array(data["predictions"]),
        np.array(data["labels"]),
    )


def pool_across_seeds(model_name: str):
    """Pool predictions across all seeds (concatenate)."""
    all_preds = []
    all_labels = []
    for seed in SEEDS:
        preds, labels = load_predictions(model_name, seed)
        all_preds.append(preds)
        all_labels.append(labels)
    return np.concatenate(all_preds), np.concatenate(all_labels)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def main():
    print("=" * 80)
    print("STATISTICAL COMPARISON: Cross-Attention vs Baselines")
    print("=" * 80)

    # Load cross-attention predictions (pooled across seeds)
    ca_preds, labels = pool_across_seeds("cross_attention")

    # Sanity check: all model predictions should have same labels order
    print(f"\nPooled test set size: {len(labels)}")

    results = {}

    for baseline in ["image_only", "metadata_only", "concat_fusion", "gated_fusion"]:
        baseline_preds, baseline_labels = pool_across_seeds(baseline)

        # Verify labels match
        assert np.array_equal(labels, baseline_labels), \
            f"Label mismatch with {baseline}"

        print(f"\n{'-'*70}")
        print(f"Cross-Attention vs {baseline}")
        print(f"{'-'*70}")

        # McNemar's test
        mc = mcnemar_test(ca_preds, baseline_preds, labels)
        print(f"  McNemar's test:")
        print(f"    A right, B wrong (b01): {mc['b01']}")
        print(f"    A wrong, B right (b10): {mc['b10']}")
        print(f"    Statistic: {mc['statistic']:.4f}")
        print(f"    p-value:   {mc['p_value']:.4g}")
        if mc['p_value'] < 0.001:
            print(f"    Result: HIGHLY SIGNIFICANT (p < 0.001)")
        elif mc['p_value'] < 0.01:
            print(f"    Result: significant (p < 0.01)")
        elif mc['p_value'] < 0.05:
            print(f"    Result: significant (p < 0.05)")
        else:
            print(f"    Result: not significant (p >= 0.05)")

        # Bootstrapped CI on Macro-F1 difference
        bs_f1 = bootstrap_metric_diff(
            ca_preds, baseline_preds, labels, macro_f1,
            n_bootstrap=5000, confidence=0.95, seed=42
        )
        print(f"\n  Bootstrap 95% CI on Macro-F1 difference:")
        print(f"    Cross-Attention F1: {bs_f1['metric_a']:.4f}")
        print(f"    {baseline} F1:      {bs_f1['metric_b']:.4f}")
        print(f"    Observed diff:      {bs_f1['observed_diff']:+.4f}")
        print(f"    95% CI:             [{bs_f1['ci_lower']:+.4f}, {bs_f1['ci_upper']:+.4f}]")
        print(f"    Bootstrap p-value:  {bs_f1['p_value']:.4g}")

        # Bootstrapped CI on Balanced Accuracy difference
        bs_ba = bootstrap_metric_diff(
            ca_preds, baseline_preds, labels, balanced_acc,
            n_bootstrap=5000, confidence=0.95, seed=42
        )
        print(f"\n  Bootstrap 95% CI on Balanced Accuracy difference:")
        print(f"    Cross-Attention:    {bs_ba['metric_a']:.4f}")
        print(f"    {baseline}:         {bs_ba['metric_b']:.4f}")
        print(f"    Observed diff:      {bs_ba['observed_diff']:+.4f}")
        print(f"    95% CI:             [{bs_ba['ci_lower']:+.4f}, {bs_ba['ci_upper']:+.4f}]")
        print(f"    Bootstrap p-value:  {bs_ba['p_value']:.4g}")

        results[f"cross_attention_vs_{baseline}"] = {
            "mcnemar": mc,
            "bootstrap_macro_f1": bs_f1,
            "bootstrap_balanced_accuracy": bs_ba,
        }

    # Save all comparisons
    out_path = RESULTS_DIR / "statistical_comparisons.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*80}")
    print(f"Saved to {out_path}")

    # Summary table for the paper
    print(f"\n{'='*80}")
    print("SUMMARY TABLE (for the paper)")
    print(f"{'='*80}")
    print(f"{'Comparison':<40} {'F1 Diff [95% CI]':<25} {'McNemar p':<12}")
    print("-" * 80)
    for name, res in results.items():
        f1_diff = res["bootstrap_macro_f1"]["observed_diff"]
        ci_lo = res["bootstrap_macro_f1"]["ci_lower"]
        ci_hi = res["bootstrap_macro_f1"]["ci_upper"]
        mc_p = res["mcnemar"]["p_value"]

        stars = ""
        if mc_p < 0.001:
            stars = "***"
        elif mc_p < 0.01:
            stars = "**"
        elif mc_p < 0.05:
            stars = "*"

        print(
            f"{name:<40} "
            f"{f1_diff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] "
            f"{mc_p:.4g}{stars}"
        )


if __name__ == "__main__":
    main()
