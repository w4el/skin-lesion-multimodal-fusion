"""Generate every numeric table in the paper from the saved result files.

No number in the results section is typed by hand. Each table is written to
tables/ as a LaTeX fragment that dissertation.tex reads with \\input, so the
paper cannot drift from the data.

Produces:
    tables/main_results.tex        main results, with parameter counts and a
                                   majority-class reference row
    tables/perclass_f1.tex         per-class F1 for every model
    tables/perclass_cross_attn.tex per-class P/R/F1 for cross-attention
    tables/ablation_direction.tex  reverse attention direction, if run
    tables/calibration.tex         ECE, if probabilities have been exported

Usage (PowerShell):
    python run_make_tables.py
    python run_make_tables.py --no_params    # skip model construction
"""

import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

import json

from src.analysis.common import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    MEL_IDX,
    SEEDS,
    available_models,
    ANALYSIS_DIR,
    ensure_dirs,
    fmt_mean_std,
    get_test_loader,
    latex_escape,
    load_predictions,
    load_probs,
    load_results,
    write_table,
)
from src.models.architectures import build_model

METRIC_KEYS = ["macro_f1", "auroc", "balanced_accuracy", "mel_sensitivity"]
METRIC_HEADERS = {
    "macro_f1": r"\textbf{Macro-F1}",
    "auroc": r"\textbf{AUROC}",
    "balanced_accuracy": r"\textbf{Balanced Acc.}",
    "mel_sensitivity": r"\textbf{Mel. Sensitivity}",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_metrics(models):
    """metrics[model][key] = list of per-seed values."""
    out = {}
    for m in models:
        out[m] = {k: [] for k in METRIC_KEYS}
        for seed in SEEDS:
            res = load_results(m, seed)["metrics"]
            for k in METRIC_KEYS:
                out[m][k].append(float(res[k]))
    return out


def count_parameters(models, metadata_dim):
    """Total parameter count per model. Backbone weights are not downloaded."""
    counts = {}
    for m in models:
        kwargs = {} if m == "metadata_only" else {"pretrained": False}
        model = build_model(m, metadata_dim=metadata_dim, num_classes=7, **kwargs)
        counts[m] = sum(p.numel() for p in model.parameters())
        del model
    return counts


def trivial_baseline(labels):
    """Macro-F1 and balanced accuracy of always predicting the majority class.

    This is the reference the metadata-only model should be compared
    against. Predicting nv for everything gives a macro-F1 near 0.11 and a
    balanced accuracy of 1/7, so a metadata-only macro-F1 of about 0.05
    combined with a balanced accuracy of about 0.34 means the model is not
    collapsing onto the majority class; it is spreading predictions across
    rare classes with very low precision.
    """
    majority = int(np.bincount(labels).argmax())
    preds = np.full_like(labels, majority)
    return {
        "majority_class": CLASS_NAMES[majority],
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mel_sensitivity": float(
            (preds[labels == MEL_IDX] == MEL_IDX).sum()
            / max(1, (labels == MEL_IDX).sum())
        ),
    }


def expected_calibration_error(probs, labels, n_bins=10):
    """Standard top-label ECE with equal-width confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_main(models, metrics, params, trivial):
    # Determine the best model per metric so bolding is data-driven.
    best = {}
    for k in METRIC_KEYS:
        means = {m: np.mean(metrics[m][k]) for m in models}
        best[k] = max(means, key=means.get)

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Test set performance across all models "
        r"(mean $\pm$ std over three seeds, best in bold). The majority-class "
        r"row predicts melanocytic nevi for every sample and is included as "
        r"the reference point for the metadata-only baseline. Parameter "
        r"counts include the shared EfficientNet-B0 backbone.}",
        r"\label{tab:main_results}",
        r"\centering",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & "
        + " & ".join(METRIC_HEADERS[k] for k in METRIC_KEYS) + r" \\",
        r"\midrule",
        f"Majority class ({trivial['majority_class']}) & -- & "
        f"{trivial['macro_f1']:.3f} & -- & "
        f"{trivial['balanced_accuracy']:.3f} & "
        f"{trivial['mel_sensitivity']:.3f} \\\\",
        r"\midrule",
    ]
    for m in models:
        p = f"{params[m]/1e6:.2f}M" if params else "--"
        cells = [
            fmt_mean_std(metrics[m][k], bold=(best[k] == m)) for k in METRIC_KEYS
        ]
        lines.append(
            f"{latex_escape(DISPLAY_NAMES[m])} & {p} & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("main_results.tex", lines)

    # Print the parameter comparison, which is an argument in the discussion.
    if params and "concat_fusion" in params and "cross_attention" in params:
        diff = params["cross_attention"] - params["concat_fusion"]
        print(f"\n  Cross-attention has {abs(diff)/1e6:.2f}M "
              f"{'more' if diff > 0 else 'fewer'} parameters than concatenation.")
        print("  If more, say so in the discussion: extra capacity, no gain.")


def table_perclass_f1(models, class_support):
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Per-class F1 score (mean $\pm$ std over three seeds). "
        r"Test-set support is shown for each class. The two rarest classes "
        r"carry very few test samples, so their per-seed variation should "
        r"not be over-interpreted.}",
        r"\label{tab:perclass_f1}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(CLASS_NAMES) + "}",
        r"\toprule",
        r"\textbf{Model} & "
        + " & ".join(rf"\textbf{{{latex_escape(c)}}}" for c in CLASS_NAMES) + r" \\",
        r"\textit{(test support)} & "
        + " & ".join(f"({class_support[c]})" for c in CLASS_NAMES) + r" \\",
        r"\midrule",
    ]
    for m in models:
        vals = {c: [] for c in CLASS_NAMES}
        for seed in SEEDS:
            rep = load_results(m, seed)["per_class_report"]
            for c in CLASS_NAMES:
                vals[c].append(float(rep[c]["f1-score"]))
        cells = [fmt_mean_std(vals[c], precision=2) for c in CLASS_NAMES]
        lines.append(
            f"{latex_escape(DISPLAY_NAMES[m])} & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("perclass_f1.tex", lines)


def table_perclass_cross_attn(class_support, model="cross_attention"):
    lines = [
        r"\begin{table}[htbp]",
        rf"\caption{{Per-class precision, recall and F1 for the "
        rf"{latex_escape(DISPLAY_NAMES[model].lower())} model "
        r"(mean $\pm$ std over three seeds).}",
        r"\label{tab:perclass_ca}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrccc}",
        r"\toprule",
        r"\textbf{Class} & \textbf{Support} & \textbf{Precision} & "
        r"\textbf{Recall} & \textbf{F1} \\",
        r"\midrule",
    ]
    for c in CLASS_NAMES:
        p, r, f = [], [], []
        for seed in SEEDS:
            rep = load_results(model, seed)["per_class_report"][c]
            p.append(float(rep["precision"]))
            r.append(float(rep["recall"]))
            f.append(float(rep["f1-score"]))
        lines.append(
            f"{latex_escape(c)} & {class_support[c]} & "
            f"{fmt_mean_std(p, 2)} & {fmt_mean_std(r, 2)} & "
            f"{fmt_mean_std(f, 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("perclass_cross_attn.tex", lines)


def table_ablation(metrics, params):
    if "cross_attention_reverse" not in metrics:
        print("  Skipping ablation table: run_ablation_reverse.py has not been run")
        return

    # Focal loss is deliberately not in available_models(), because it changes
    # the objective and does not belong in the main results table. Load it
    # here so it can appear as a third ablation row.
    has_focal = False
    if "cross_attention_focal" not in metrics:
        try:
            vals = {k: [] for k in METRIC_KEYS}
            for seed in SEEDS:
                res = load_results("cross_attention_focal", seed)["metrics"]
                for k in METRIC_KEYS:
                    vals[k].append(float(res[k]))
            metrics["cross_attention_focal"] = vals
            has_focal = True
            print("  Focal loss results found, adding third ablation row")
        except FileNotFoundError:
            print("  No focal loss results; ablation table will have two rows")
    else:
        has_focal = True
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Ablations on the cross-attention model (mean $\pm$ std "
        r"over three seeds). Reversed direction swaps queries with keys and "
        r"values, so three metadata tokens attend over 49 image patches "
        r"rather than the reverse; the parameter count is unchanged. Focal "
        r"loss replaces the class-weighted cross-entropy term, with the "
        r"weighted sampler retained, so it is a single-variable change of "
        r"objective rather than an additional imbalance correction. All "
        r"other settings are identical in every row.}",
        r"\label{tab:ablation_direction}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Configuration} & "
        + " & ".join(METRIC_HEADERS[k] for k in METRIC_KEYS) + r" \\",
        r"\midrule",
    ]
    rows = [
        ("Primary (image queries, weighted CE)", "cross_attention"),
        ("Reversed attention direction", "cross_attention_reverse"),
    ]
    if has_focal:
        rows.append(("Focal loss objective", "cross_attention_focal"))
    for label, m in rows:
        cells = [fmt_mean_std(metrics[m][k]) for k in METRIC_KEYS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("ablation_direction.tex", lines)


def table_leakage(metrics):
    """Concatenation against cross-attention under both split protocols.

    The point of the table is the last column: how the gap between the two
    models changes when lesion-level leakage is reintroduced.
    """
    pairs = [("concat_fusion", "concat_fusion_leaky"),
             ("cross_attention", "cross_attention_leaky")]
    leaky = {}
    for grouped_name, leaky_name in pairs:
        vals = {k: [] for k in METRIC_KEYS}
        try:
            for seed in SEEDS:
                res = load_results(leaky_name, seed)["metrics"]
                for k in METRIC_KEYS:
                    vals[k].append(float(res[k]))
        except FileNotFoundError:
            print("  Skipping leakage table: run_leakage_experiment.py "
                  "has not completed")
            return
        leaky[leaky_name] = vals

    stats_path = ANALYSIS_DIR.parent / "leakage_overlap_stats.json"
    pct = None
    if stats_path.exists():
        with open(stats_path) as f:
            pct = json.load(f).get("pct_test_contaminated")

    caption = (
        r"Concatenation and cross-attention under two evaluation protocols "
        r"(mean $\pm$ std over three seeds). The grouped split assigns all "
        r"images of a lesion to one partition; the non-grouped split is "
        r"matched on partition sizes and class distribution and differs only "
        r"in assigning images independently")
    if pct is not None:
        caption += (rf", so {pct}\% of test images share a lesion with "
                    r"training")
    caption += (r". The final row gives the difference between the two "
                r"models under each protocol.}")

    lines = [
        r"\begin{table}[htbp]",
        r"\caption{" + caption,
        r"\label{tab:leakage}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Model} & \textbf{Grouped} & "
        r"\textbf{Non-grouped} \\",
        r"\midrule",
    ]
    for k in ("macro_f1", "balanced_accuracy", "auroc"):
        label = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced Acc.",
                 "auroc": "AUROC"}[k]
        lines.append(
            f"\\multirow{{3}}{{*}}{{{label}}} & Concatenation & "
            f"{fmt_mean_std(metrics['concat_fusion'][k])} & "
            f"{fmt_mean_std(leaky['concat_fusion_leaky'][k])} \\\\")
        lines.append(
            f" & Cross-Attention & "
            f"{fmt_mean_std(metrics['cross_attention'][k])} & "
            f"{fmt_mean_std(leaky['cross_attention_leaky'][k])} \\\\")
        g = np.mean(metrics['cross_attention'][k]) - np.mean(metrics['concat_fusion'][k])
        l = np.mean(leaky['cross_attention_leaky'][k]) - np.mean(leaky['concat_fusion_leaky'][k])
        lines.append(
            f" & \\textit{{Difference}} & ${g:+.3f}$ & ${l:+.3f}$ \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    write_table("leakage_comparison.tex", lines)

    print("\n  LEAKAGE EFFECT (cross-attention minus concatenation):")
    for k in ("macro_f1", "balanced_accuracy", "auroc"):
        g = np.mean(metrics['cross_attention'][k]) - np.mean(metrics['concat_fusion'][k])
        l = np.mean(leaky['cross_attention_leaky'][k]) - np.mean(leaky['concat_fusion_leaky'][k])
        verdict = "WIDENS under leakage" if l > g else "does not widen"
        print(f"    {k:<20} grouped {g:+.4f} -> non-grouped {l:+.4f}   {verdict}")


def table_calibration(models):
    rows = []
    for m in models:
        eces = []
        for seed in SEEDS:
            try:
                probs = load_probs(m, seed)
            except FileNotFoundError:
                return print("  Skipping calibration table: "
                             "run run_export_probabilities.py first")
            _, labels = load_predictions(m, seed)
            eces.append(expected_calibration_error(probs, labels))
        rows.append((m, eces))

    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Expected calibration error (10 equal-width confidence "
        r"bins, top-label, mean $\pm$ std over three seeds). Lower is "
        r"better. Calibration measures whether predicted confidence matches "
        r"observed accuracy and is independent of ranking quality.}",
        r"\label{tab:calibration}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{ECE} \\",
        r"\midrule",
    ]
    best = min(rows, key=lambda r: np.mean(r[1]))[0]
    for m, eces in rows:
        lines.append(
            f"{latex_escape(DISPLAY_NAMES[m])} & "
            f"{fmt_mean_std(eces, bold=(m == best))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("calibration.tex", lines)
    print(f"  Best calibrated: {DISPLAY_NAMES[best]}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_params", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    models = available_models()
    print(f"Models: {', '.join(models)}")

    metrics = collect_metrics(models)

    _, labels = load_predictions(models[0], SEEDS[0])
    trivial = trivial_baseline(labels)
    print(f"\nMajority-class reference ({trivial['majority_class']}): "
          f"macro-F1 {trivial['macro_f1']:.4f}, "
          f"balanced accuracy {trivial['balanced_accuracy']:.4f}")
    print("Compare the metadata-only row against this before writing "
          "anything about 'close to the majority-class prior'.")

    class_support = {
        c: int((labels == i).sum()) for i, c in enumerate(CLASS_NAMES)
    }
    print(f"Test-set support: {class_support}")

    params = None
    if not args.no_params:
        print("\nCounting parameters...")
        _, _, _, encoder = get_test_loader(num_workers=0)
        params = count_parameters(models, encoder.metadata_dim)
        for m in models:
            print(f"  {DISPLAY_NAMES[m]:<28} {params[m]:>12,}")

    print("\nGenerating tables...")
    table_main(models, metrics, params, trivial)
    table_perclass_f1(models, class_support)
    table_perclass_cross_attn(class_support)
    table_ablation(metrics, params)
    table_leakage(metrics)
    table_calibration(models)
    print("\nDone.")


if __name__ == "__main__":
    main()