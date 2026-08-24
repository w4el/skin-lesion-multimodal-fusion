"""Write placeholders so the revised paper compiles before results exist.

The revised dissertation.tex reads its results tables with \\input and
includes three figures that are produced later in the workflow. Without
placeholders it will not compile, so the page count cannot be measured.

This writes:
  - eight stub tables with the correct structure, dash-filled
  - three placeholder figures at the right aspect ratio

Nothing existing is overwritten. If a real table or figure is already in
place, it is left alone, so this is safe to re-run at any point.

Every placeholder is visibly marked, so one cannot be mistaken for a
result in a compiled PDF.

Usage (PowerShell):
    python make_stubs.py
"""

from pathlib import Path

TABLES_DIR = Path("tables")
FIGURES_DIR = Path("figures")
DASH = "--"
NOTE = r"\textcolor{red}{PLACEHOLDER. Regenerate before submission.}"

MODELS = [
    "Metadata-Only MLP", "Image-Only CNN", "Concatenation",
    "Gated Fusion", "Cross-Attention",
]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BASELINES = ["Metadata-Only MLP", "Image-Only CNN", "Concatenation", "Gated Fusion"]


def write_table(name, lines):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / name
    if path.exists():
        print(f"  {path} exists, left alone")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("% PLACEHOLDER. Overwritten by the generator scripts.\n")
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote stub {path}")


def write_placeholder_figure(name, width, height, caption):
    """Create a marked placeholder image at roughly the right shape."""
    path = FIGURES_DIR / name
    if path.exists():
        print(f"  {path} exists, left alone")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  Could not create {path}: matplotlib not available")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_facecolor("#eeeeee")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cc0000")
        spine.set_linewidth(2)
    ax.text(0.5, 0.58, "PLACEHOLDER", ha="center", va="center",
            fontsize=22, weight="bold", color="#cc0000", transform=ax.transAxes)
    ax.text(0.5, 0.40, caption, ha="center", va="center",
            fontsize=11, color="#555555", transform=ax.transAxes)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote placeholder {path}")


def main():
    print("Tables:")

    write_table("main_results.tex", [
        r"\begin{table*}[t]",
        rf"\caption{{Test set performance across all models. {NOTE}}}",
        r"\label{tab:main_results}", r"\centering",
        r"\begin{tabular}{lrcccc}", r"\toprule",
        r"\textbf{Model} & \textbf{Params} & \textbf{Macro-F1} & "
        r"\textbf{AUROC} & \textbf{Balanced Acc.} & \textbf{Mel. Sens.} \\",
        r"\midrule",
        rf"Majority class (nv) & {DASH} & {DASH} & {DASH} & {DASH} & {DASH} \\",
        r"\midrule",
        *[f"{m} & {DASH} & {DASH} & {DASH} & {DASH} & {DASH} \\\\" for m in MODELS],
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ])

    write_table("statistical_pooled.tex", [
        r"\begin{table}[htbp]",
        rf"\caption{{Pooled statistical comparison. {NOTE}}}",
        r"\label{tab:statistical_pooled}", r"\centering", r"\small",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"\textbf{Comparison} & $b_{01}$ & $b_{10}$ & \textbf{McNemar $p$} & "
        r"\textbf{Favours} & \textbf{Macro-F1 diff} \\",
        r"\midrule",
        *[f"CA vs {b} & {DASH} & {DASH} & {DASH} & {DASH} & {DASH} \\\\"
          for b in BASELINES],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    write_table("statistical_perseed.tex", [
        r"\begin{table}[htbp]",
        rf"\caption{{Per-seed statistical comparison. {NOTE}}}",
        r"\label{tab:statistical_perseed}", r"\centering", r"\small",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"\multirow{2}{*}{\textbf{Comparison}} & "
        r"\multicolumn{2}{c}{\textbf{Seed 42}} & "
        r"\multicolumn{2}{c}{\textbf{Seed 123}} & "
        r"\multicolumn{2}{c}{\textbf{Seed 456}} \\",
        r" & $p$ & $\Delta$F1 & $p$ & $\Delta$F1 & $p$ & $\Delta$F1 \\",
        r"\midrule",
        *[f"CA vs {b} & " + " & ".join([DASH] * 6) + r" \\" for b in BASELINES],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    write_table("perclass_f1.tex", [
        r"\begin{table*}[t]",
        rf"\caption{{Per-class F1 score. {NOTE}}}",
        r"\label{tab:perclass_f1}", r"\centering", r"\small",
        r"\begin{tabular}{l" + "c" * len(CLASSES) + "}", r"\toprule",
        r"\textbf{Model} & " + " & ".join(rf"\textbf{{{c}}}" for c in CLASSES) + r" \\",
        r"\textit{(test support)} & " + " & ".join([DASH] * len(CLASSES)) + r" \\",
        r"\midrule",
        *[f"{m} & " + " & ".join([DASH] * len(CLASSES)) + r" \\" for m in MODELS],
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ])

    write_table("perclass_cross_attn.tex", [
        r"\begin{table}[htbp]",
        rf"\caption{{Per-class precision, recall and F1, cross-attention. {NOTE}}}",
        r"\label{tab:perclass_ca}", r"\centering", r"\small",
        r"\begin{tabular}{lrccc}", r"\toprule",
        r"\textbf{Class} & \textbf{Support} & \textbf{Precision} & "
        r"\textbf{Recall} & \textbf{F1} \\", r"\midrule",
        *[f"{c} & {DASH} & {DASH} & {DASH} & {DASH} \\\\" for c in CLASSES],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    write_table("ablation_direction.tex", [
        r"\begin{table}[htbp]",
        rf"\caption{{Attention direction ablation. {NOTE}}}",
        r"\label{tab:ablation_direction}", r"\centering", r"\small",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Direction} & \textbf{Macro-F1} & \textbf{AUROC} & "
        r"\textbf{Balanced Acc.} & \textbf{Mel. Sens.} \\", r"\midrule",
        rf"Image queries (primary) & {DASH} & {DASH} & {DASH} & {DASH} \\",
        rf"Metadata queries (reversed) & {DASH} & {DASH} & {DASH} & {DASH} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    write_table("calibration.tex", [
        r"\begin{table}[htbp]",
        rf"\caption{{Expected calibration error. {NOTE}}}",
        r"\label{tab:calibration}", r"\centering", r"\small",
        r"\begin{tabular}{lc}", r"\toprule",
        r"\textbf{Model} & \textbf{ECE} \\", r"\midrule",
        *[f"{m} & {DASH} \\\\" for m in MODELS],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    write_table("melanoma_sensitivity.tex", [
        r"\begin{table}[htbp]",
        rf"\caption{{Melanoma sensitivity, gated fusion vs cross-attention. {NOTE}}}",
        r"\label{tab:mel_sensitivity}", r"\centering", r"\small",
        r"\begin{tabular}{lc}", r"\toprule",
        r"\textbf{Quantity} & \textbf{Value} \\", r"\midrule",
        rf"Discordant melanoma cases & {DASH} \\",
        rf"Detected by one model only ($b_{{01}}$ / $b_{{10}}$) & {DASH} \\",
        rf"McNemar exact $p$ & {DASH} \\",
        rf"Sensitivity difference [95\% CI] & {DASH} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    print("\nFigures:")
    write_placeholder_figure("confusion_matrices.png", 7.5, 5.0,
                             "Confusion matrices")
    write_placeholder_figure("roc_melanoma.png", 6.0, 5.5,
                             "Melanoma ROC curves")
    write_placeholder_figure("calibration.png", 6.0, 5.5,
                             "Reliability diagrams")

    print("\nDone. The paper will now compile.")
    print("Placeholders are marked in red and cannot pass for results.")


if __name__ == "__main__":
    main()
