"""Statistical comparison utilities for model evaluation.

Provides McNemar's test and bootstrapped confidence intervals to
determine whether performance differences between models are
statistically reliable.
"""

from typing import Dict

import numpy as np
from scipy import stats


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test on paired correctness for two models.

    Counts b01 (A correct, B wrong) and b10 (A wrong, B correct) and tests
    whether those two counts differ. A significant result means the models
    differ in plain accuracy, in the direction of the larger count. It does
    not say anything about the models making different *kinds* of error.
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # Contingency: b wrong & a right vs a wrong & b right
    b01 = np.sum(correct_a & ~correct_b)  # A right, B wrong
    b10 = np.sum(~correct_a & correct_b)  # A wrong, B right

    # McNemar's test with continuity correction
    if b01 + b10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b01": int(b01), "b10": int(b10)}

    statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "b01": int(b01),
        "b10": int(b10),
    }


def bootstrap_metric_diff(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrapped confidence interval for the difference in a
    metric between two models.

    Args:
        preds_a: Predictions from model A.
        preds_b: Predictions from model B.
        labels: Ground truth labels.
        metric_fn: Function(labels, preds) -> float.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dictionary with observed difference, CI bounds, and p-value.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)

    observed_a = metric_fn(labels, preds_a)
    observed_b = metric_fn(labels, preds_b)
    observed_diff = observed_a - observed_b

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = metric_fn(labels[idx], preds_a[idx]) - metric_fn(
            labels[idx], preds_b[idx]
        )
        diffs.append(diff)

    diffs = np.array(diffs)
    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    # Two-sided p-value: proportion of bootstrap samples on opposite side of 0
    p_value = np.mean(diffs <= 0) if observed_diff > 0 else np.mean(diffs >= 0)
    p_value = 2 * min(p_value, 1 - p_value)  # two-sided

    return {
        "observed_diff": float(observed_diff),
        "metric_a": float(observed_a),
        "metric_b": float(observed_b),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "confidence": confidence,
    }


def compare_all_models(
    predictions: Dict[str, np.ndarray],
    labels: np.ndarray,
    reference_model: str = "cross_attention",
) -> Dict[str, Dict]:
    """Compare a reference model against all others using McNemar's
    test and bootstrapped F1 confidence intervals.

    Args:
        predictions: Dict mapping model names to prediction arrays.
        labels: Ground truth labels.
        reference_model: Name of the model to compare against.

    Returns:
        Dictionary of comparison results.
    """
    from sklearn.metrics import f1_score

    def macro_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    ref_preds = predictions[reference_model]
    results = {}

    for name, preds in predictions.items():
        if name == reference_model:
            continue

        results[f"{reference_model}_vs_{name}"] = {
            "mcnemar": mcnemar_test(ref_preds, preds, labels),
            "bootstrap_f1": bootstrap_metric_diff(
                ref_preds, preds, labels, macro_f1
            ),
        }

    return results