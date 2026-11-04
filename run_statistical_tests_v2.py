"""Paired statistical comparison of cross-attention against each baseline.

Three properties matter for how the results can be read.

1. Direction is reported. McNemar's test compares paired correctness, so a
   significant result means the two models differ in plain accuracy, and
   the sign of (b01 - b10) says which model is ahead. A significant
   p-value on its own says nothing about which model is better, and it is
   not evidence that the models make different kinds of error. Every
   comparison prints the favoured model, and the generated tables carry
   b01 and b10 as columns.

2. Tests are run per seed as well as pooled. Pooling three seeds gives
   4,482 rows, but those are the same 1,494 test images counted three
   times. The tests treat them as independent, which inflates the effective
   sample size roughly threefold and shrinks p-values. Per-seed results are
   the defensible primary analysis; pooled results are reported as a
   secondary, optimistic estimate.

3. The bootstrap two-sided p-value is computed as
   2 * min(P(diff <= 0), P(diff >= 0)), clipped to 1, so neither tail is
   double-counted.

An exact binomial test on the discordant pairs is reported alongside the
chi-square approximation, because several discordant counts are small.

Outputs:
    results/analysis/statistical_comparisons_v2.json
    tables/statistical_pooled.tex
    tables/statistical_perseed.tex
    tables/melanoma_sensitivity.tex
    A summary of the key comparisons printed at the end.

Usage (PowerShell):
    python run_statistical_tests_v2.py
"""

import json
from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.analysis.common import (
    ANALYSIS_DIR,
    DISPLAY_NAMES,
    MEL_IDX,
    SEEDS,
    available_models,
    ensure_dirs,
    fmt_ci,
    fmt_p,
    latex_escape,
    load_predictions,
    write_table,
)

REFERENCE = "cross_attention"
N_BOOTSTRAP = 5000


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def mel_recall(y_true, y_pred):
    mask = y_true == MEL_IDX
    if mask.sum() == 0:
        return 0.0
    return float((y_pred[mask] == MEL_IDX).sum() / mask.sum())


METRICS = {
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_acc,
    "mel_sensitivity": mel_recall,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def mcnemar(preds_a, preds_b, labels, name_a, name_b):
    """McNemar's test with direction made explicit."""
    correct_a = preds_a == labels
    correct_b = preds_b == labels

    b01 = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    b10 = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n_disc = b01 + b10

    if n_disc == 0:
        return {
            "b01": 0, "b10": 0, "n_discordant": 0,
            "statistic": 0.0, "p_value": 1.0, "p_exact": 1.0,
            "favoured": "neither", "acc_a": float(correct_a.mean()),
            "acc_b": float(correct_b.mean()),
        }

    # Chi-square with continuity correction, matching the original script.
    statistic = (abs(b01 - b10) - 1) ** 2 / n_disc
    p_chi2 = float(1 - stats.chi2.cdf(statistic, df=1))

    # Exact binomial on the discordant pairs. Preferred when n_disc is small.
    p_exact = float(stats.binomtest(b01, n_disc, 0.5).pvalue)

    if b01 > b10:
        favoured = name_a
    elif b10 > b01:
        favoured = name_b
    else:
        favoured = "neither"

    return {
        "b01": b01,
        "b10": b10,
        "n_discordant": n_disc,
        "statistic": float(statistic),
        "p_value": p_chi2,
        "p_exact": p_exact,
        "favoured": favoured,
        "acc_a": float(correct_a.mean()),
        "acc_b": float(correct_b.mean()),
    }


def bootstrap_diff(preds_a, preds_b, labels, metric_fn,
                   n_bootstrap=N_BOOTSTRAP, confidence=0.95, seed=42):
    """Paired bootstrap CI on a metric difference (A minus B)."""
    rng = np.random.RandomState(seed)
    n = len(labels)

    observed_a = metric_fn(labels, preds_a)
    observed_b = metric_fn(labels, preds_b)
    observed_diff = observed_a - observed_b

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = (
            metric_fn(labels[idx], preds_a[idx])
            - metric_fn(labels[idx], preds_b[idx])
        )

    alpha = 1 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    p_low = float(np.mean(diffs <= 0))
    p_high = float(np.mean(diffs >= 0))
    p_value = min(1.0, 2 * min(p_low, p_high))

    return {
        "metric_a": float(observed_a),
        "metric_b": float(observed_b),
        "observed_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "crosses_zero": bool(ci_lower <= 0 <= ci_upper),
    }


def mel_subset_mcnemar(preds_a, preds_b, labels, name_a, name_b):
    """McNemar restricted to melanoma cases.

    This is the paired test for a difference in melanoma sensitivity: on the
    melanoma-labelled samples only, correctness is exactly 'was this
    melanoma detected'.
    """
    mask = labels == MEL_IDX
    return mcnemar(preds_a[mask], preds_b[mask], labels[mask], name_a, name_b)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def pool(model_name):
    all_p, all_l = [], []
    for seed in SEEDS:
        p, l = load_predictions(model_name, seed)
        all_p.append(p)
        all_l.append(l)
    return np.concatenate(all_p), np.concatenate(all_l)


def compare(preds_a, preds_b, labels, name_a, name_b, boot_seed=42):
    out = {"mcnemar": mcnemar(preds_a, preds_b, labels, name_a, name_b)}
    for key, fn in METRICS.items():
        out[f"bootstrap_{key}"] = bootstrap_diff(
            preds_a, preds_b, labels, fn, seed=boot_seed
        )
    return out


def main():
    ensure_dirs()
    models = available_models()
    baselines = [m for m in models if m != REFERENCE]

    print("=" * 78)
    print("STATISTICAL COMPARISON v2: Cross-Attention vs baselines")
    print("=" * 78)
    print(f"Models found: {', '.join(models)}")

    results = {"per_seed": {}, "pooled": {}, "melanoma": {}}

    # ---------------- Per-seed ----------------
    for seed in SEEDS:
        ca_preds, labels = load_predictions(REFERENCE, seed)
        results["per_seed"][str(seed)] = {}
        print(f"\n{'#'*78}\nSEED {seed}  (n = {len(labels)})\n{'#'*78}")

        for baseline in baselines:
            b_preds, b_labels = load_predictions(baseline, seed)
            assert np.array_equal(labels, b_labels), f"Label mismatch: {baseline}"

            res = compare(ca_preds, b_preds, labels, REFERENCE, baseline)
            results["per_seed"][str(seed)][baseline] = res

            mc = res["mcnemar"]
            f1 = res["bootstrap_macro_f1"]
            print(f"\n  Cross-Attention vs {baseline}")
            print(f"    accuracy: {mc['acc_a']:.4f} vs {mc['acc_b']:.4f}")
            print(f"    b01 (CA right, base wrong): {mc['b01']}")
            print(f"    b10 (CA wrong, base right): {mc['b10']}")
            print(f"    McNemar chi2 p = {mc['p_value']:.4g}  "
                  f"exact p = {mc['p_exact']:.4g}  favours: {mc['favoured']}")
            print(f"    Macro-F1 diff {f1['observed_diff']:+.4f} "
                  f"[{f1['ci_lower']:+.4f}, {f1['ci_upper']:+.4f}] "
                  f"p = {f1['p_value']:.4g}")

    # ---------------- Pooled ----------------
    print(f"\n{'#'*78}\nPOOLED ACROSS SEEDS (not independent, read as optimistic)"
          f"\n{'#'*78}")
    ca_pooled, labels_pooled = pool(REFERENCE)
    print(f"Pooled n = {len(labels_pooled)} "
          f"({len(labels_pooled)//len(SEEDS)} images x {len(SEEDS)} seeds)")

    for baseline in baselines:
        b_pooled, b_labels = pool(baseline)
        assert np.array_equal(labels_pooled, b_labels), f"Label mismatch: {baseline}"

        res = compare(ca_pooled, b_pooled, labels_pooled, REFERENCE, baseline)
        results["pooled"][baseline] = res

        mc = res["mcnemar"]
        print(f"\n  Cross-Attention vs {baseline}")
        print(f"    b01 = {mc['b01']}, b10 = {mc['b10']}, "
              f"p = {mc['p_value']:.4g}, favours: {mc['favoured']}")
        for key in METRICS:
            bs = res[f"bootstrap_{key}"]
            print(f"    {key:<18} {bs['observed_diff']:+.4f} "
                  f"[{bs['ci_lower']:+.4f}, {bs['ci_upper']:+.4f}] "
                  f"p = {bs['p_value']:.4g}")

    # ---------------- Melanoma sensitivity ----------------
    print(f"\n{'#'*78}\nMELANOMA SENSITIVITY, PAIRWISE\n{'#'*78}")
    print("McNemar restricted to melanoma-labelled test cases.")

    for a, b in combinations(models, 2):
        per_seed_p = []
        for seed in SEEDS:
            pa, la = load_predictions(a, seed)
            pb, lb = load_predictions(b, seed)
            per_seed_p.append(mel_subset_mcnemar(pa, pb, la, a, b))

        pa_pool, l_pool = pool(a)
        pb_pool, _ = pool(b)
        pooled = mel_subset_mcnemar(pa_pool, pb_pool, l_pool, a, b)
        boot = bootstrap_diff(pa_pool, pb_pool, l_pool, mel_recall)

        results["melanoma"][f"{a}_vs_{b}"] = {
            "per_seed": per_seed_p, "pooled": pooled, "bootstrap_pooled": boot,
        }

        if {a, b} == {"gated_fusion", "cross_attention"}:
            marker = "   <-- the comparison claimed in the discussion"
        else:
            marker = ""
        print(f"\n  {a} vs {b}{marker}")
        print(f"    pooled: {pooled['b01']} / {pooled['b10']} discordant, "
              f"exact p = {pooled['p_exact']:.4g}, favours: {pooled['favoured']}")
        print(f"    sensitivity diff {boot['observed_diff']:+.4f} "
              f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]")

    # ---------------- Save ----------------
    out_path = ANALYSIS_DIR / "statistical_comparisons_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    build_tables(results, baselines, models)
    print_summary(results, baselines)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def build_tables(results, baselines, models):
    print("\nGenerating LaTeX tables...")

    # Pooled table with direction visible
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Cross-attention (CA) against each baseline on pooled test "
        r"predictions. $b_{01}$ counts cases where CA is correct and the "
        r"baseline is not; $b_{10}$ the reverse. McNemar's test compares "
        r"these counts, so a significant result indicates a difference in "
        r"plain accuracy in the direction of the larger count. Bootstrap "
        r"95\% CIs use 5{,}000 resamples. Pooled predictions repeat the same "
        r"1{,}494 test images across three seeds and are therefore not "
        r"independent; see Table~\ref{tab:statistical_perseed}. "
        r"Significance: * $p<0.05$, ** $p<0.01$, *** $p<0.001$.}",
        r"\label{tab:statistical_pooled}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Comparison} & $b_{01}$ & $b_{10}$ & \textbf{McNemar $p$} "
        r"& \textbf{Favours} & \textbf{Macro-F1 diff [95\% CI]} \\",
        r"\midrule",
    ]
    for b in baselines:
        mc = results["pooled"][b]["mcnemar"]
        f1 = results["pooled"][b]["bootstrap_macro_f1"]
        fav = "CA" if mc["favoured"] == REFERENCE else (
            "baseline" if mc["favoured"] != "neither" else "neither")
        lines.append(
            f"CA vs {latex_escape(DISPLAY_NAMES[b])} & {mc['b01']} & {mc['b10']} & "
            f"{fmt_p(mc['p_value'])} & {fav} & "
            f"{fmt_ci(f1['observed_diff'], f1['ci_lower'], f1['ci_upper'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("statistical_pooled.tex", lines)

    # Per-seed table
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Per-seed statistical comparison of cross-attention against "
        r"each baseline. Each seed is an independent evaluation on the same "
        r"1{,}494-image test set. McNemar $p$ is the chi-square "
        r"approximation with continuity correction; the sign of the "
        r"macro-F1 difference is CA minus baseline.}",
        r"\label{tab:statistical_perseed}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "ccc" * len(SEEDS) + "}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Comparison}} & "
        + " & ".join([r"\multicolumn{3}{c}{\textbf{Seed " + str(s) + "}}"
                      for s in SEEDS]) + r" \\",
        " & " + " & ".join([r"$p$ & $\Delta$F1 & favours" for _ in SEEDS]) + r" \\",
        r"\midrule",
    ]
    for b in baselines:
        cells = []
        for seed in SEEDS:
            r = results["per_seed"][str(seed)][b]
            cells.append(fmt_p(r["mcnemar"]["p_value"]))
            cells.append(f"${r['bootstrap_macro_f1']['observed_diff']:+.3f}$")
            fav = r["mcnemar"]["favoured"]
            cells.append("CA" if fav == REFERENCE else
                         ("base" if fav != "neither" else "--"))
        lines.append(
            f"CA vs {latex_escape(DISPLAY_NAMES[b])} & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table("statistical_perseed.tex", lines)

    # Melanoma sensitivity table
    key = None
    for k in results["melanoma"]:
        if "gated_fusion" in k and "cross_attention" in k and "reverse" not in k:
            key = k
            break
    if key:
        entry = results["melanoma"][key]
        mc, boot = entry["pooled"], entry["bootstrap_pooled"]
        lines = [
            r"\begin{table}[htbp]",
            r"\caption{Melanoma sensitivity, gated fusion against "
            r"cross-attention. McNemar is restricted to melanoma-labelled "
            r"test cases, where correctness is equivalent to detection. "
            r"Bootstrap CI is on the difference in melanoma recall.}",
            r"\label{tab:mel_sensitivity}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"\textbf{Quantity} & \textbf{Value} \\",
            r"\midrule",
            f"Discordant melanoma cases & {mc['n_discordant']} \\\\",
            f"Detected by one model only ($b_{{01}}$ / $b_{{10}}$) & "
            f"{mc['b01']} / {mc['b10']} \\\\",
            f"McNemar exact $p$ & {fmt_p(mc['p_exact'])} \\\\",
            f"Sensitivity difference [95\\% CI] & "
            f"{fmt_ci(boot['observed_diff'], boot['ci_lower'], boot['ci_upper'])} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        write_table("melanoma_sensitivity.tex", lines)


# ---------------------------------------------------------------------------
# Summary of the comparisons that carry the argument
# ---------------------------------------------------------------------------

def print_summary(results, baselines):
    print("\n" + "=" * 78)
    print("KEY COMPARISONS")
    print("All values below are computed from the stored predictions.")
    print("=" * 78)

    concat = results["pooled"].get("concat_fusion")
    if concat:
        mc = concat["mcnemar"]
        f1 = concat["bootstrap_macro_f1"]
        ba = concat["bootstrap_balanced_accuracy"]
        direction = (
            "concatenation is the more accurate of the two"
            if mc["favoured"] == "concat_fusion"
            else "cross-attention is the more accurate of the two"
        )
        print(f"""
[Cross-attention vs concatenation]
On pooled predictions, cross-attention and concatenation disagree on
{mc['n_discordant']} test cases. Cross-attention is correct on {mc['b01']} of
these and concatenation on {mc['b10']}, so under McNemar's test
{direction} (p = {mc['p_value']:.3g}). The difference in macro-F1 is
{f1['observed_diff']:+.3f} with a 95% confidence interval of
[{f1['ci_lower']:+.3f}, {f1['ci_upper']:+.3f}], and in balanced accuracy
{ba['observed_diff']:+.3f} [{ba['ci_lower']:+.3f}, {ba['ci_upper']:+.3f}].
""")

    print("""
[Reconciling McNemar with macro-F1]
McNemar's test and the macro-F1 interval measure different things, which is
why they can point in different directions. McNemar operates on plain
paired correctness, which on HAM10000 is dominated by melanocytic nevi at
roughly two thirds of all samples, so a model that is stronger on the
majority class can be significantly more accurate overall. Macro-F1 weights
all seven classes equally, so gains concentrated in nv are largely
invisible to it. The two results together indicate that the models differ
mainly in how they handle the majority class rather than in overall
per-class balance.
""")

    print("""
[Why pooled p-values are optimistic]
Pooled tests treat 4,482 predictions as independent observations, but they
comprise the same 1,494 test images evaluated under three random seeds.
This inflates the effective sample size and shrinks p-values accordingly.
Per-seed results are therefore reported alongside the pooled analysis, and
the pooled p-values should be read as optimistic.
""")


if __name__ == "__main__":
    main()