"""Grad-CAM support for the two-input fusion models.

The pytorch-grad-cam library expects model(image) -> logits, but every
fusion model here takes model(image, metadata). ModelWrapper holds the
metadata fixed so the library sees the single-input signature it needs.
"""

import torch.nn as nn


def get_target_layer(model, model_name: str):
    """Return the convolutional layer Grad-CAM should hook.

    Both supported models wrap the same EfficientNet-B0, but at different
    attribute depths, so the path differs.
    """
    if model_name == "image_only":
        return model.backbone.backbone.conv_head
    elif model_name == "cross_attention":
        # Same backbone, taken before the patch tokens are extracted
        return model.image_backbone.backbone.conv_head
    else:
        raise ValueError(f"Grad-CAM not supported for {model_name}")


class ModelWrapper(nn.Module):
    """Expose a two-input fusion model as model(image) -> logits.

    Metadata is fixed at construction. Grad-CAM then attributes only to
    the image, which is what the spatial heatmap is meant to show.
    """

    def __init__(self, model, metadata):
        super().__init__()
        self.model = model
        self.metadata = metadata

    def forward(self, image):
        # Grad-CAM may pass a batch even when metadata is a single row
        batch_size = image.shape[0]
        if self.metadata.shape[0] != batch_size:
            metadata = self.metadata.expand(batch_size, -1)
        else:
            metadata = self.metadata
        return self.model(image, metadata)
