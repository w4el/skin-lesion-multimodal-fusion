"""Leakage-controlled data splitting for HAM10000.

Implements a lesion_id-grouped stratified split so that all images of the
same lesion stay in the same partition. This prevents lesion-level data
leakage between train, validation and test sets.

NOTE: This prevents lesion-level leakage but does NOT guarantee patient-level
leakage prevention, since HAM10000 does not provide a reliable patient
identifier.
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def create_splits(
    metadata_path: str,
    output_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, list]:
    """Create lesion_id-grouped stratified train/val/test splits.

    Groups by lesion_id so that multiple images of the same lesion never
    appear across different partitions. Uses a two-stage GroupShuffleSplit
    to achieve a three-way split while preserving approximate class
    stratification.

    Args:
        metadata_path: Path to HAM10000_metadata.csv.
        output_path: Path to save the split indices JSON file.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to lists of
        image_id strings.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    df = pd.read_csv(metadata_path)

    # Deduplicate to lesion level for splitting purposes.
    # Each lesion_id gets assigned to one split; all its images follow.
    lesion_df = df.drop_duplicates(subset="lesion_id")[
        ["lesion_id", "dx"]
    ].reset_index(drop=True)

    # Stage 1: split into train vs (val + test)
    holdout_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=holdout_ratio, random_state=seed
    )

    # Use lesion_id as both the group and the index
    train_lesion_idx, holdout_lesion_idx = next(
        gss1.split(lesion_df, lesion_df["dx"], groups=lesion_df["lesion_id"])
    )

    holdout_lesion_df = lesion_df.iloc[holdout_lesion_idx].reset_index(drop=True)

    # Stage 2: split holdout into val vs test
    relative_test_ratio = test_ratio / holdout_ratio
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=relative_test_ratio, random_state=seed
    )

    val_lesion_idx, test_lesion_idx = next(
        gss2.split(
            holdout_lesion_df,
            holdout_lesion_df["dx"],
            groups=holdout_lesion_df["lesion_id"],
        )
    )

    # Collect lesion_ids for each split
    train_lesions = set(lesion_df.iloc[train_lesion_idx]["lesion_id"])
    val_lesions = set(holdout_lesion_df.iloc[val_lesion_idx]["lesion_id"])
    test_lesions = set(holdout_lesion_df.iloc[test_lesion_idx]["lesion_id"])

    # Verify zero overlap
    assert len(train_lesions & val_lesions) == 0, "Train-val lesion overlap!"
    assert len(train_lesions & test_lesions) == 0, "Train-test lesion overlap!"
    assert len(val_lesions & test_lesions) == 0, "Val-test lesion overlap!"

    # Map back to image_ids
    splits = {
        "train": df[df["lesion_id"].isin(train_lesions)]["image_id"].tolist(),
        "val": df[df["lesion_id"].isin(val_lesions)]["image_id"].tolist(),
        "test": df[df["lesion_id"].isin(test_lesions)]["image_id"].tolist(),
    }

    # Print split statistics
    print(f"\nSplit statistics (seed={seed}):")
    print(f"  Total images: {len(df)}")
    print(f"  Train: {len(splits['train'])} images, {len(train_lesions)} lesions")
    print(f"  Val:   {len(splits['val'])} images, {len(val_lesions)} lesions")
    print(f"  Test:  {len(splits['test'])} images, {len(test_lesions)} lesions")

    # Print class distribution per split
    print("\nClass distribution per split:")
    for split_name, image_ids in splits.items():
        split_df = df[df["image_id"].isin(image_ids)]
        dist = split_df["dx"].value_counts(normalize=True).sort_index()
        print(f"  {split_name}:")
        for cls_name, ratio in dist.items():
            count = (split_df["dx"] == cls_name).sum()
            print(f"    {cls_name}: {count} ({ratio:.1%})")

    # Save splits
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSplits saved to {output_path}")

    return splits


def load_splits(split_path: str) -> Dict[str, list]:
    """Load previously saved split indices.

    Args:
        split_path: Path to the splits JSON file.

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to lists of
        image_id strings.
    """
    with open(split_path, "r") as f:
        return json.load(f)