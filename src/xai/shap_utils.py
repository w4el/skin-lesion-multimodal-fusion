"""KernelSHAP support for the two-input fusion models.

KernelSHAP perturbs a single feature vector, so the image branch has to
be held constant. MetadataOnlyWrapper fixes the image and exposes
metadata as the only varying input, taking and returning numpy so it
plugs straight into shap.KernelExplainer.
"""

import numpy as np
import torch
import torch.nn as nn


class MetadataOnlyWrapper(nn.Module):
    """Expose a fusion model as f(metadata) -> class probabilities.

    The image is fixed at construction, so SHAP attributes only to the
    metadata features.
    """

    def __init__(self, model, fixed_image: torch.Tensor):
        super().__init__()
        self.model = model
        self.fixed_image = fixed_image  # (1, 3, 224, 224)

    def forward(self, metadata_np: np.ndarray) -> np.ndarray:
        """Take (N, 20) numpy metadata, return (N, 7) numpy probabilities."""
        device = next(self.model.parameters()).device
        metadata = torch.tensor(metadata_np, dtype=torch.float32).to(device)
        n = metadata.shape[0]

        # Chunked because broadcasting one image to N > 100 rows overflows
        # PyTorch's 32-bit indexing on the backbone
        chunk_size = 16
        all_probs = []
        with torch.no_grad():
            for i in range(0, n, chunk_size):
                meta_chunk = metadata[i:i + chunk_size]
                image_chunk = self.fixed_image.expand(
                    meta_chunk.shape[0], -1, -1, -1
                ).to(device)
                logits = self.model(image_chunk, meta_chunk)
                probs = torch.softmax(logits.float(), dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)
