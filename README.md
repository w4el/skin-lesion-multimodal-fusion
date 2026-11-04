# Cross-Attention Multimodal Fusion for Skin Lesion Classification on HAM10000

Code and results for the MSc dissertation of the same title, awarded a
distinction (75%).

Wael Al Mobarak
MSc Computer Science (Data Science), Queen Mary University of London
Supervisor: Dr Syed Rafee

---

## Results at a glance

Test set, mean ± std over three seeds, lesion-level grouped split:

| Model | Macro-F1 | AUROC | Balanced Acc. | Mel. Sensitivity |
|---|---|---|---|---|
| Metadata-only MLP | 0.051 ± 0.001 | 0.721 ± 0.018 | 0.336 ± 0.006 | 0.000 |
| Image-only CNN | 0.642 ± 0.035 | 0.944 ± 0.006 | 0.692 ± 0.018 | 0.777 ± 0.018 |
| Concatenation | **0.675 ± 0.008** | 0.946 ± 0.002 | **0.760 ± 0.016** | 0.774 ± 0.048 |
| Gated fusion | 0.641 ± 0.038 | 0.941 ± 0.004 | 0.726 ± 0.041 | **0.809 ± 0.009** |
| Cross-attention | 0.663 ± 0.028 | **0.954 ± 0.003** | 0.747 ± 0.019 | 0.758 ± 0.009 |

- Cross-attention gives the best AUROC but does not beat simple
  concatenation on macro-F1 or balanced accuracy, and per-seed tests
  disagree on the sign of the difference.
- Retraining both on a matched split that reintroduces lesion-level leakage
  (35% of test images share a lesion with training) inflates cross-attention
  more (+0.080 macro-F1 against +0.061) and reverses the ordering.
- Reversing the attention direction changes nothing measurable; focal loss
  raises macro-F1 to 0.715 but drops melanoma sensitivity to 0.665.

---

## 1. What this is

A controlled comparison of five strategies for combining dermoscopic images
with clinical metadata on HAM10000: an image-only CNN, a metadata-only MLP,
concatenation fusion, gated fusion and cross-attention fusion. All five
share an ImageNet-pretrained EfficientNet-B0 backbone, one identical
training protocol and a single fixed lesion-level grouped split, so any
difference between them is attributable to the fusion mechanism.

Three further experiments extend that comparison: an ablation reversing the
cross-attention direction, an ablation swapping class-weighted
cross-entropy for focal loss, and an experiment that retrains two models on
a deliberately non-grouped split to measure the effect of lesion-level
leakage.

---

## 2. Guide to the files

```
.
├── src/                             importable package
│   ├── data/
│   │   ├── split.py                 lesion_id-grouped stratified split
│   │   ├── metadata.py              age, sex and localisation encoding
│   │   └── dataset.py               Dataset and DataLoader factory
│   ├── models/
│   │   └── architectures.py         all five models plus build_model()
│   ├── training/
│   │   ├── trainer.py               training loop and multi-seed runner
│   │   └── statistics.py            McNemar and bootstrap
│   ├── analysis/
│   │   └── common.py                full-precision evaluation, shared helpers
│   ├── utils/
│   │   └── seed.py                  seeding and determinism flags
│   └── xai/
│       ├── gradcam.py               single-input wrapper for Grad-CAM
│       └── shap_utils.py            metadata-only wrapper for KernelSHAP
│
├── run_experiments.py               trains the five primary models, 3 seeds
├── recompute_metrics.py             recomputes metrics in full precision
├── run_ablation_reverse.py          reversed attention direction
├── run_focal_loss_ablation.py       focal loss instead of weighted CE
├── run_leakage_experiment.py        non-grouped split, retrains two models
├── run_export_probabilities.py      caches test probabilities to results/probs/
├── run_statistical_tests_v2.py      per-seed and pooled paired tests
├── run_make_tables.py               writes every LaTeX table in tables/
├── run_make_figures.py              confusion matrices, ROC, reliability
├── run_make_architecture_figure.py  the five-panel schematic
├── run_gradcam.py                   Grad-CAM heatmaps
├── run_shap.py                      KernelSHAP metadata attributions
│
├── data/HAM10000_metadata.csv       dataset metadata (images not included)
├── results/                         all JSON results, splits, probabilities
├── tables/                          generated .tex fragments used by the paper
├── figures/                         generated .png figures used by the paper
└── environment.yml                  conda environment
```

Model checkpoints (`results/*.pth`, about 570MB across 27 runs) are not
included. Every number in the paper can be regenerated without them, as
described in section 5.

---

## 3. Why there is no executable

This is a research pipeline rather than an application. It has no user
interface and nothing to package into a single binary. Reproducing the full
set of results requires the HAM10000 image set, roughly 6GB, which cannot be
redistributed here, and a CUDA GPU for about 20 hours of training.

Section 4 lists the steps to run it from source. Section 5 describes what
can be reproduced in about a minute, on a CPU, with no dataset download.

---

## 4. Running from source

### 4.1 Environment

```powershell
conda env create -f environment.yml
conda activate skinlesion
```

**Then verify CUDA, because the environment file does not guarantee it:**

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, pip has installed a CPU-only build. `environment.yml`
requests `torch>=2.0` without a CUDA index URL, so the wheel selected depends
on the platform. Fix it with:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 4.2 Dataset

Download HAM10000 from Harvard Dataverse:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Place it so the layout is:

```
data/
├── HAM10000_metadata.csv        (already included here)
├── HAM10000_images_part_1/      (5,000 .jpg files)
└── HAM10000_images_part_2/      (5,015 .jpg files)
```

Note that `run_experiments.py` resolves these two image directories by an
explicit path lookup rather than a recursive search, so the folder names
above must match exactly.

### 4.3 Order of execution

```powershell
python run_experiments.py                # 5 models x 3 seeds, about 12 GPU-hours
python recompute_metrics.py              # full-precision metrics from checkpoints
python run_ablation_reverse.py           # about 2 GPU-hours
python run_leakage_experiment.py         # about 4 GPU-hours
python run_focal_loss_ablation.py        # about 1.5 GPU-hours
python run_export_probabilities.py       # minutes
python run_statistical_tests_v2.py       # minutes
python run_make_tables.py                # seconds
python run_make_figures.py               # seconds
python run_gradcam.py                    # minutes
python run_shap.py                       # minutes
```

`recompute_metrics.py` is not optional. AUROC was computed inside the
mixed-precision context during training, where float16 probability rows do
not sum to exactly 1.0 and scikit-learn rejects them. Every metric reported
in the paper is recomputed in full precision from the saved best
checkpoints, not taken from the training logs.

---

## 5. Reproducing the paper's numbers without a GPU

The stored predictions are included, so every table and figure in the paper
can be regenerated on a CPU in under a minute, with no dataset download and
no checkpoints:

```powershell
conda activate skinlesion
python run_make_tables.py
python run_statistical_tests_v2.py
python run_make_figures.py
```

`run_make_tables.py` also reports parameter counts, which it obtains by
constructing each model. That makes `timm` fetch the ImageNet-pretrained
EfficientNet-B0 weights, so this one script needs an internet connection on
first run. The weights are cached afterwards. The other two scripts read
only local files.

These read only `results/*.json` and `results/probs/*.npy`. The regenerated
files in `tables/` are exactly what the dissertation source reads with
`\input`, so no number in the results section is typed by hand and the paper
cannot drift from the data. The `.tex` and `.bib` sources are submitted
separately with the dissertation itself.

---

## 6. Which script produces which output

| Paper item | Produced by |
|---|---|
| Table II, main results | `run_make_tables.py` |
| Table III, ablations | `run_make_tables.py` |
| Table IV, leakage comparison | `run_make_tables.py` |
| Table V, per-class F1 | `run_make_tables.py` |
| Table VI, per-class cross-attention | `run_make_tables.py` |
| Table VII, calibration | `run_make_tables.py` |
| Table VIII, pooled statistics | `run_statistical_tests_v2.py` |
| Table IX, per-seed statistics | `run_statistical_tests_v2.py` |
| Table X, melanoma sensitivity | `run_statistical_tests_v2.py` |
| Figure 1, architectures | `run_make_architecture_figure.py` |
| Figure 2, confusion matrices | `run_make_figures.py` |
| Figure 3, reliability diagrams | `run_make_figures.py` |
| Figure 4, SHAP heatmap | `run_shap.py` |
| Figures 5 and 6, Grad-CAM | `run_gradcam.py` |

Table I and Table XI are written by hand in the dissertation source.

---

## 7. Hardware and runtimes

All training was run on a single NVIDIA RTX 4060 Ti (16GB), Windows 11,
Python 3.10, PyTorch with CUDA 12.4. Individual runs took between 15 and 80
minutes depending on the model and on early stopping. The full set of 27
training runs comes to roughly 20 GPU-hours.

Seeds are set for Python, NumPy and PyTorch, with `cudnn.deterministic`
enabled and `cudnn.benchmark` disabled. DataLoaders use four workers without
a per-worker seed function, so batch composition is not bit-for-bit
reproducible between runs. The seed-to-seed standard deviations reported in
the paper therefore include a small non-deterministic component alongside
genuine initialisation variance.

---

## 8. Data licence

HAM10000 is released under CC BY-NC 4.0 and is not redistributed here. Only
`HAM10000_metadata.csv` is included, so that the split and the metadata
encoding can be inspected without downloading the images.
Attribution: Tschandl, P., Rosendahl, C. and Kittler, H. (2018), "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", Scientific Data 5, 180161.
