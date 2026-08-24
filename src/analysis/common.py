"""Shared helpers for the post-hoc analysis scripts.

Everything in this module is read-only with respect to the original
experiment outputs. No script that imports this module overwrites any
existing file in results/. New outputs go to results/analysis/,
results/probs/, tables/ and figures/.

It also registers the reverse-direction cross-attention variant in the
model registry so that build_model() and any checkpoint-globbing script
can resolve the name 'cross_attention_reverse'.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import torch

from src.data.dataset import create_dataloaders
from src.data.metadata import DIAGNOSIS_LABELS, DIAGNOSIS_NAMES
from src.data.split import load_splits
from src.models import architectures
from src.models.architectures import CrossAttentionFusion, build_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
ANALYSIS_DIR = RESULTS_DIR / "analysis"
PROBS_DIR = RESULTS_DIR / "probs"
TABLES_DIR = Path("tables")
FIGURES_DIR = Path("figures")

METADATA_PATH = "data/HAM10000_metadata.csv"
IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2",
]
SPLITS_PATH = "results/splits.json"

SEEDS = [42, 123, 456]

# Presentation order used in every table and figure.
MODELS = [
    "metadata_only",
    "image_only",
    "concat_fusion",
    "gated_fusion",
    "cross_attention",
]

# Included only if the ablation has been run.
ABLATION_MODELS = ["cross_attention_reverse"]

DISPLAY_NAMES = {
    "metadata_only": "Metadata-Only MLP",
    "image_only": "Image-Only CNN",
    "concat_fusion": "Concatenation",
    "gated_fusion": "Gated Fusion",
    "cross_attention": "Cross-Attention",
    "cross_attention_reverse": "Cross-Attention (reversed)",
}

CLASS_NAMES = [DIAGNOSIS_NAMES[i] for i in range(len(DIAGNOSIS_LABELS))]
MEL_IDX = DIAGNOSIS_LABELS["mel"]


# ---------------------------------------------------------------------------
# Reverse-direction ablation variant
# ---------------------------------------------------------------------------

class CrossAttentionFusionReverse(CrossAttentionFusion):
    """Cross-attention with metadata tokens as queries and image patches as
    keys and values.

    This is the ablation of the directionality argument made in the
    methodology. It is a thin subclass rather than a flag on the runner so
    that its checkpoints and result files get their own filenames and cannot
    overwrite the primary cross-attention run.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("reverse_direction", None)
        super().__init__(*args, reverse_direction=True, **kwargs)


architectures.MODEL_REGISTRY["cross_attention_reverse"] = CrossAttentionFusionReverse

# ---------------------------------------------------------------------------
# Leakage experiment aliases
# ---------------------------------------------------------------------------
# The leakage experiment retrains two existing architectures on a
# deliberately non-grouped split. The architectures are unchanged, so these
# are registry aliases rather than new classes. Separate names keep their
# checkpoints and result files away from the primary runs.

for _base, _alias in (("concat_fusion", "concat_fusion_leaky"),
                      ("cross_attention", "cross_attention_leaky")):
    if _base in architectures.MODEL_REGISTRY:
        architectures.MODEL_REGISTRY[_alias] = architectures.MODEL_REGISTRY[_base]

LEAKY_MODELS = ["concat_fusion_leaky", "cross_attention_leaky"]

SPLITS_LEAKY_PATH = "results/splits_leaky.json"

DISPLAY_NAMES["concat_fusion_leaky"] = "Concatenation (leaky split)"
DISPLAY_NAMES["cross_attention_leaky"] = "Cross-Attention (leaky split)"



# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_test_loader(batch_size: int = 32, num_workers: int = 0):
    """Build the test DataLoader and metadata encoder.

    num_workers defaults to 0 so that evaluation order is deterministic.
    """
    splits = load_splits(SPLITS_PATH)
    train_loader, val_loader, test_loader, encoder = create_dataloaders(
        metadata_path=METADATA_PATH,
        image_dirs=IMAGE_DIRS,
        splits=splits,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, encoder


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fp32(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference in full precision.

    Mixed precision is deliberately not used here. Softmax probabilities
    computed under autocast were the cause of the AUROC failure in the
    original training runs, so every number used in the paper is produced
    by this function.

    Returns:
        preds, labels, probs
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, metadata, labels in loader:
        images = images.to(device)
        metadata = metadata.to(device)
        logits = model(images, metadata).float()
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def load_checkpoint_model(model_name: str, seed: int, metadata_dim: int, device):
    """Build a model and load its best checkpoint for a given seed."""
    ckpt = RESULTS_DIR / f"{model_name}_seed{seed}_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model = build_model(model_name, metadata_dim=metadata_dim, num_classes=7).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Results IO
# ---------------------------------------------------------------------------

def results_path(model_name: str, seed: int) -> Path:
    return RESULTS_DIR / f"{model_name}_seed{seed}_test_results.json"


def load_results(model_name: str, seed: int) -> Dict:
    with open(results_path(model_name, seed)) as f:
        return json.load(f)


def load_predictions(model_name: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    data = load_results(model_name, seed)
    return np.array(data["predictions"]), np.array(data["labels"])


def probs_path(model_name: str, seed: int) -> Path:
    return PROBS_DIR / f"{model_name}_seed{seed}_probs.npy"


def load_probs(model_name: str, seed: int) -> np.ndarray:
    path = probs_path(model_name, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_export_probabilities.py first."
        )
    return np.load(path)


def available_models(include_ablation: bool = True) -> List[str]:
    """Return the models that actually have result files for every seed."""
    candidates = MODELS + (ABLATION_MODELS if include_ablation else [])
    found = []
    for m in candidates:
        if all(results_path(m, s).exists() for s in SEEDS):
            found.append(m)
        elif m in ABLATION_MODELS:
            continue
        else:
            print(f"  WARNING: incomplete results for {m}, skipping")
    return found


def ensure_dirs():
    for d in (ANALYSIS_DIR, PROBS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def latex_escape(text: str) -> str:
    return (
        text.replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def fmt_mean_std(values, precision: int = 3, bold: bool = False) -> str:
    """Format a list of per-seed values as mean +/- std."""
    m, s = float(np.mean(values)), float(np.std(values))
    cell = f"{m:.{precision}f} $\\pm$ {s:.{precision}f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def fmt_ci(diff: float, lo: float, hi: float, precision: int = 3) -> str:
    return (
        f"${diff:+.{precision}f}$ "
        f"[${lo:+.{precision}f}$, ${hi:+.{precision}f}$]"
    )


def fmt_p(p: float) -> str:
    """Format a p-value with significance stars."""
    if p < 1e-3:
        base = "$<0.001$"
    elif p < 0.01:
        base = f"${p:.4f}$"
    else:
        base = f"${p:.3f}$"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return base + stars


def write_table(filename: str, lines: List[str]) -> None:
    """Write a generated LaTeX fragment to tables/."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    header = [
        "% Generated file. Do not edit by hand.",
        "% Regenerate with the corresponding run_make_*.py script.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines) + "\n")
    print(f"  Wrote {path}")
