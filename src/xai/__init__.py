"""Interpretability helpers shared by the Grad-CAM and SHAP scripts."""

from src.xai.gradcam import ModelWrapper, get_target_layer
from src.xai.shap_utils import MetadataOnlyWrapper

__all__ = ["ModelWrapper", "get_target_layer", "MetadataOnlyWrapper"]