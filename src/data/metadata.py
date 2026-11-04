"""Metadata encoding for HAM10000 clinical metadata.

Encodes the limited clinical metadata available in HAM10000 (age, sex,
localization) into numerical tensors suitable for model input. Age is
z-score normalised using training set statistics only. Sex and localization
are one-hot encoded.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch


# Canonical orderings for consistent one-hot encoding
SEX_CATEGORIES = ["male", "female", "unknown"]

LOCALIZATION_CATEGORIES = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
    "upper extremity", "unknown",
]

# Diagnosis label mapping
DIAGNOSIS_LABELS = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

DIAGNOSIS_NAMES = {v: k for k, v in DIAGNOSIS_LABELS.items()}


class MetadataEncoder:
    """Encodes HAM10000 metadata fields into a single tensor.

    Computes normalisation statistics from the training set only and
    applies them consistently to val/test sets to prevent data leakage.

    Attributes:
        age_mean: Training set mean age (computed during fit).
        age_std: Training set std age (computed during fit).
        age_median: Training set median age (for imputing missing values).
        metadata_dim: Total dimensionality of the encoded metadata vector.
    """

    def __init__(self):
        self.age_mean: Optional[float] = None
        self.age_std: Optional[float] = None
        self.age_median: Optional[float] = None
        # 1 (age) + 1 (age_missing flag) + len(SEX) + len(LOCALIZATION)
        self.metadata_dim = 1 + 1 + len(SEX_CATEGORIES) + len(LOCALIZATION_CATEGORIES)

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        """Compute normalisation statistics from training data only.

        Args:
            df: Training set metadata DataFrame.

        Returns:
            self, for method chaining.
        """
        ages = df["age"].dropna()
        self.age_mean = ages.mean()
        self.age_std = ages.std()
        self.age_median = ages.median()

        if self.age_std == 0 or np.isnan(self.age_std):
            self.age_std = 1.0

        return self

    def encode(self, row: pd.Series) -> torch.Tensor:
        """Encode a single metadata row into a tensor.

        Args:
            row: A row from the HAM10000 metadata DataFrame.

        Returns:
            Float tensor of shape (metadata_dim,).
        """
        assert self.age_mean is not None, "Must call fit() before encode()"

        features = []

        # Age: z-score normalisation with missing indicator
        if pd.isna(row["age"]):
            features.append(0.0)  # imputed normalised value
            features.append(1.0)  # missing flag
        else:
            features.append((row["age"] - self.age_mean) / self.age_std)
            features.append(0.0)  # not missing

        # Sex: one-hot
        sex = row["sex"] if pd.notna(row["sex"]) else "unknown"
        sex_onehot = [1.0 if sex == cat else 0.0 for cat in SEX_CATEGORIES]
        features.extend(sex_onehot)

        # Localization: one-hot
        loc = row["localization"] if pd.notna(row["localization"]) else "unknown"
        loc_onehot = [1.0 if loc == cat else 0.0 for cat in LOCALIZATION_CATEGORIES]
        features.extend(loc_onehot)

        return torch.tensor(features, dtype=torch.float32)

    def encode_dataframe(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Encode all rows in a DataFrame.

        Args:
            df: Metadata DataFrame.

        Returns:
            Dictionary mapping image_id to encoded metadata tensor.
        """
        encoded = {}
        for _, row in df.iterrows():
            encoded[row["image_id"]] = self.encode(row)
        return encoded