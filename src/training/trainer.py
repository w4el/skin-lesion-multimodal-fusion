"""Training loop, evaluation, and multi-seed experiment runner.

Provides a unified training and evaluation pipeline that all five model
variants plug into identically, ensuring fair comparison.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.data.metadata import DIAGNOSIS_LABELS, DIAGNOSIS_NAMES
from src.utils.seed import set_seed


class Trainer:
    """Unified trainer for all model variants.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularisation.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience (epochs without val improvement).
        class_weights: Optional tensor of class weights for loss function.
        use_focal_loss: If True, use focal loss instead of cross-entropy.
        focal_gamma: Gamma parameter for focal loss.
        save_dir: Directory to save checkpoints and logs.
        experiment_name: Name for this experiment run.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        patience: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        save_dir: str = "results",
        experiment_name: str = "experiment",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Mixed precision
        self.scaler = GradScaler()

        # Tracking
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0
        self.history: List[Dict] = []

    def _focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss for handling class imbalance.

        Args:
            logits: Raw model outputs (B, num_classes).
            targets: Ground truth labels (B,).

        Returns:
            Scalar loss.
        """
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()

    def train_one_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, metadata, labels in self.train_loader:
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda"):
                logits = self.model(images, metadata)
                if self.use_focal_loss:
                    loss = self._focal_loss(logits, labels)
                else:
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the model on a DataLoader.

        Returns:
            Tuple of (metrics_dict, all_preds, all_labels, all_probs).
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for images, metadata, labels in loader:
            images = images.to(self.device)
            metadata = metadata.to(self.device)

            with autocast("cuda"):
                logits = self.model(images, metadata)

            probs = torch.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.float().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Compute metrics
        metrics = {}
        metrics["macro_f1"] = f1_score(all_labels, all_preds, average="macro")
        metrics["balanced_accuracy"] = balanced_accuracy_score(
            all_labels, all_preds
        )

        # One-vs-rest AUROC
        try:
            metrics["auroc"] = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            )
        except Exception as e:
            print(f"AUROC error: {type(e).__name__}: {e}")
            metrics["auroc"] = 0.0

        # Melanoma sensitivity (recall for mel class)
        mel_idx = DIAGNOSIS_LABELS["mel"]
        mel_mask = all_labels == mel_idx
        if mel_mask.sum() > 0:
            metrics["mel_sensitivity"] = (
                (all_preds[mel_mask] == mel_idx).sum() / mel_mask.sum()
            )
        else:
            metrics["mel_sensitivity"] = 0.0

        return metrics, all_preds, all_labels, all_probs

    def train(self) -> Dict:
        """Run the full training loop with early stopping.

        Returns:
            Dictionary containing training history and best metrics.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.save_dir / f"{self.experiment_name}_best.pth"

        print(f"\nTraining {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_metrics, _, _, _ = self.evaluate(self.val_loader)

            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_data)

            print(
                f"  Epoch {epoch+1:3d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f} | "
                f"Val AUROC: {val_metrics['auroc']:.4f} | "
                f"Val Bal.Acc: {val_metrics['balanced_accuracy']:.4f} | "
                f"Mel Sens: {val_metrics['mel_sensitivity']:.4f}"
            )

            # Early stopping check
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - start_time
        print(f"  Training completed in {elapsed/60:.1f} minutes")
        print(f"  Best val Macro-F1: {self.best_val_f1:.4f}")

        # Load best model for final evaluation
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        return {
            "history": self.history,
            "best_val_f1": self.best_val_f1,
            "training_time_minutes": elapsed / 60,
        }


def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels.

    Args:
        train_loader: Training DataLoader.

    Returns:
        Tensor of class weights.
    """
    labels = train_loader.dataset.get_labels()
    class_counts = np.bincount(labels, minlength=len(DIAGNOSIS_LABELS))
    total = len(labels)
    weights = total / (len(DIAGNOSIS_LABELS) * class_counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    experiment_name: str,
) -> Dict:
    """Run full evaluation on the test set and save results.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        device: Torch device.
        save_dir: Directory to save results.
        experiment_name: Name for this experiment.

    Returns:
        Dictionary of test metrics.
    """
    model.to(device)
    model.eval()

    trainer = Trainer(
        model=model,
        train_loader=test_loader,  # unused, just to satisfy init
        val_loader=test_loader,
        device=device,
    )

    metrics, preds, labels, probs = trainer.evaluate(test_loader)

    # Per-class report
    class_names = [DIAGNOSIS_NAMES[i] for i in range(len(DIAGNOSIS_LABELS))]
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )
    conf_mat = confusion_matrix(labels, preds)

    results = {
        "metrics": metrics,
        "per_class_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
        "probabilities": probs.tolist(),
    }

    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"{experiment_name}_test_results.json", "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "probabilities"},
            f,
            indent=2,
            default=str,
        )

    print(f"\nTest Results for {experiment_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=class_names))
    print(f"Confusion matrix:\n{conf_mat}")

    return results


def run_multi_seed(
    model_name: str,
    model_kwargs: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seeds: List[int] = [42, 123, 456, 789, 1024],
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 10,
    use_focal_loss: bool = False,
    save_dir: str = "results",
) -> Dict:
    """Run the same model configuration across multiple seeds.

    Args:
        model_name: Name of the model to build.
        model_kwargs: Keyword arguments for build_model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        device: Torch device.
        seeds: List of random seeds.
        num_epochs: Max epochs per run.
        learning_rate: Learning rate.
        patience: Early stopping patience.
        use_focal_loss: Whether to use focal loss.
        save_dir: Directory to save results.

    Returns:
        Dictionary with per-seed results and aggregated statistics.
    """
    from src.models.architectures import build_model

    all_metrics = []
    all_preds = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed} / {len(seeds)}")
        print(f"{'='*60}")

        set_seed(seed)

        # Build fresh model for each seed
        model = build_model(model_name, **model_kwargs)

        # Get class weights
        class_weights = get_class_weights(train_loader)

        experiment_name = f"{model_name}_seed{seed}"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            save_dir=save_dir,
            experiment_name=experiment_name,
        )

        trainer.train()

        # Test evaluation
        results = evaluate_on_test(
            model, test_loader, device, save_dir, experiment_name
        )

        all_metrics.append(results["metrics"])
        all_preds.append(results["predictions"])

    # Aggregate across seeds
    metric_keys = all_metrics[0].keys()
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    summary = {
        "model_name": model_name,
        "seeds": seeds,
        "per_seed_metrics": all_metrics,
        "aggregated": aggregated,
    }

    # Save summary
    summary_path = Path(save_dir) / f"{model_name}_multi_seed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Multi-seed summary for {model_name}:")
    for key, vals in aggregated.items():
        print(f"  {key}: {vals['mean']:.4f} +/- {vals['std']:.4f}")
    print(f"{'='*60}")

    return summary