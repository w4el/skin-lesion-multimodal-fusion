"""All five model architectures for the comparative study.

1. ImageOnlyCNN: EfficientNet-B0 backbone with classification head.
2. MetadataOnlyMLP: MLP classifier using only clinical metadata.
3. ConcatFusion: Late fusion via feature concatenation.
4. GatedFusion: Late fusion with a learned gating mechanism.
5. CrossAttentionFusion: Cross-attention between image patches and metadata.

All models share the same backbone and hyperparameter structure where
applicable, to isolate the effect of the fusion strategy.
"""

from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

class ImageBackbone(nn.Module):
    """EfficientNet-B0 backbone with configurable output.

    Can return either pooled features (1280-dim vector) for late fusion
    models, or unpooled patch features (49 x 1280) for cross-attention.

    Args:
        pretrained: Whether to load ImageNet pretrained weights.
        return_patches: If True, return spatial feature map before GAP.
    """

    def __init__(self, pretrained: bool = True, return_patches: bool = False):
        super().__init__()
        self.return_patches = return_patches
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        self.feature_dim = 1280  # EfficientNet-B0 feature dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_patches:
            # Get feature map before global average pooling
            # timm's forward_features returns (B, C, H, W)
            features = self.backbone.forward_features(x)
            B, C, H, W = features.shape
            # Reshape to (B, H*W, C) = (B, 49, 1280) for 224x224 input
            features = features.reshape(B, C, H * W).permute(0, 2, 1)
            return features
        else:
            # Global average pooled features: (B, 1280)
            return self.backbone(x)


class MetadataMLP(nn.Module):
    """MLP encoder for clinical metadata.

    Args:
        input_dim: Dimensionality of the encoded metadata vector.
        hidden_dim: Hidden layer size.
        output_dim: Output embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Model 1: Image-Only CNN
# ---------------------------------------------------------------------------

class ImageOnlyCNN(nn.Module):
    """Image-only baseline using EfficientNet-B0.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights.
        dropout: Dropout before final classifier.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = ImageBackbone(pretrained=pretrained, return_patches=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(
        self, image: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass. Metadata is accepted but ignored."""
        features = self.backbone(image)
        features = self.dropout(features)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Model 2: Metadata-Only MLP
# ---------------------------------------------------------------------------

class MetadataOnlyMLP(nn.Module):
    """Metadata-only baseline using a simple MLP.

    Args:
        metadata_dim: Dimensionality of encoded metadata.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = MetadataMLP(metadata_dim, hidden_dim, 64, dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(
        self, image: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass. Image is accepted but ignored."""
        features = self.mlp(metadata)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Model 3: Concatenation Fusion
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """Late fusion by concatenating image and metadata features.

    Args:
        metadata_dim: Dimensionality of encoded metadata.
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained backbone weights.
        fusion_hidden: Hidden dimension of the fusion head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 7,
        pretrained: bool = True,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_backbone = ImageBackbone(pretrained=pretrained, return_patches=False)
        self.meta_mlp = MetadataMLP(metadata_dim, 128, 64, dropout)

        fused_dim = self.image_backbone.feature_dim + self.meta_mlp.output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(
        self, image: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        img_features = self.image_backbone(image)  # (B, 1280)
        meta_features = self.meta_mlp(metadata)     # (B, 64)
        fused = torch.cat([img_features, meta_features], dim=1)  # (B, 1344)
        return self.fusion_head(fused)


# ---------------------------------------------------------------------------
# Model 4: Gated Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """Late fusion with a learned gating mechanism.

    The gate learns to weight the contribution of image vs metadata
    features dynamically per sample.

    Args:
        metadata_dim: Dimensionality of encoded metadata.
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained backbone weights.
        fusion_dim: Common dimension to project both modalities into.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 7,
        pretrained: bool = True,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_backbone = ImageBackbone(pretrained=pretrained, return_patches=False)
        self.meta_mlp = MetadataMLP(metadata_dim, 128, 64, dropout)

        # Project both modalities to the same dimension
        self.img_proj = nn.Linear(self.image_backbone.feature_dim, fusion_dim)
        self.meta_proj = nn.Linear(self.meta_mlp.output_dim, fusion_dim)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self, image: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        img_features = self.img_proj(self.image_backbone(image))   # (B, fusion_dim)
        meta_features = self.meta_proj(self.meta_mlp(metadata))     # (B, fusion_dim)

        combined = torch.cat([img_features, meta_features], dim=1)  # (B, 2*fusion_dim)
        g = self.gate(combined)  # (B, fusion_dim), values in [0, 1]

        fused = g * img_features + (1 - g) * meta_features  # (B, fusion_dim)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Model 5: Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between image patches and metadata.

    Image patch features serve as queries, and metadata embeddings serve
    as keys and values. This allows each spatial region of the image to
    attend to clinical context, which is the natural direction: metadata
    modulates how the model interprets the image.

    Args:
        metadata_dim: Dimensionality of encoded metadata.
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained backbone weights.
        d_model: Internal dimension for attention.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention layers.
        dropout: Dropout rate.
        reverse_direction: If True, swap Q and K/V (ablation study).
    """

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 7,
        pretrained: bool = True,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
        reverse_direction: bool = False,
    ):
        super().__init__()
        self.reverse_direction = reverse_direction

        # Image branch: extract patch features
        self.image_backbone = ImageBackbone(
            pretrained=pretrained, return_patches=True
        )
        self.img_proj = nn.Linear(self.image_backbone.feature_dim, d_model)

        # Metadata branch: project each metadata group as a separate token
        # This gives cross-attention more to work with than a single vector
        self.meta_mlp = MetadataMLP(metadata_dim, 128, d_model, dropout)
        # We also create per-field tokens for richer attention
        # age(2) + sex(3) + localization(15) = 20 input features
        # Group into 3 tokens: age, sex, localization
        self.age_proj = nn.Linear(2, d_model)       # age + missing flag
        self.sex_proj = nn.Linear(3, d_model)        # one-hot sex
        self.loc_proj = nn.Linear(15, d_model)       # one-hot localization

        # Cross-attention layers
        self.attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.norm_layers.append(nn.LayerNorm(d_model))
            self.ff_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
            )
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, image: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        # Image patches: (B, 49, d_model)
        img_tokens = self.img_proj(self.image_backbone(image))

        # Metadata tokens: (B, 3, d_model)
        age_token = self.age_proj(metadata[:, :2]).unsqueeze(1)      # (B, 1, d_model)
        sex_token = self.sex_proj(metadata[:, 2:5]).unsqueeze(1)     # (B, 1, d_model)
        loc_token = self.loc_proj(metadata[:, 5:20]).unsqueeze(1)    # (B, 1, d_model)
        meta_tokens = torch.cat([age_token, sex_token, loc_token], dim=1)  # (B, 3, d_model)

        # Cross-attention: image attends to metadata (or reverse)
        if self.reverse_direction:
            query, key_value = meta_tokens, img_tokens
        else:
            query, key_value = img_tokens, meta_tokens

        x = query
        for attn, norm, ff in zip(
            self.attention_layers, self.norm_layers, self.ff_layers
        ):
            # Cross-attention with residual
            attn_out, _ = attn(x, key_value, key_value)
            x = norm(x + attn_out)
            # Feedforward with residual
            x = x + ff(x)

        x = self.final_norm(x)

        # Global average pool over token dimension
        x = x.mean(dim=1)  # (B, d_model)

        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "image_only": ImageOnlyCNN,
    "metadata_only": MetadataOnlyMLP,
    "concat_fusion": ConcatFusion,
    "gated_fusion": GatedFusion,
    "cross_attention": CrossAttentionFusion,
}


def build_model(
    model_name: str,
    metadata_dim: int,
    num_classes: int = 7,
    **kwargs,
) -> nn.Module:
    """Build a model by name.

    Args:
        model_name: One of 'image_only', 'metadata_only', 'concat_fusion',
                     'gated_fusion', 'cross_attention'.
        metadata_dim: Dimensionality of encoded metadata.
        num_classes: Number of output classes.
        **kwargs: Additional model-specific arguments.

    Returns:
        An nn.Module instance.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_name]

    if model_name == "image_only":
        return cls(num_classes=num_classes, **kwargs)
    elif model_name == "metadata_only":
        return cls(metadata_dim=metadata_dim, num_classes=num_classes, **kwargs)
    else:
        return cls(
            metadata_dim=metadata_dim, num_classes=num_classes, **kwargs
        )
