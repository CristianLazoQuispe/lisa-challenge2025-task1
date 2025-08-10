"""
Models for the LISA 2025 challenge.

This module provides a timm‑based 2D model (`Model2DTimm`) with optional
axis information integration.  It implements two head types: a simple
independent per‑label classifier (`MultiHeadSimple`) and a more
sophisticated head with label tokens and a lightweight transformer
encoder (`LabelTokenHead`).

Compared to the original code provided by the user, this version adds
support for concatenating a one‑hot view vector (axial/coronal/sagittal)
to the trunk features via an embedding layer.  If ``use_view`` is
enabled, the view embedding is added to the backbone output before
passing it to the classification head.  This allows the model to learn
view‑specific patterns, which has been shown to improve classification
accuracy【42949289359295†L366-L374】.

To create the backbone, any architecture available in timm can be used
via the `base_model` argument.  By default the backbone has no
classification head (``num_classes=0``) so that features are extracted
directly.
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict

import torch
import torch.nn as nn
import timm


class MultiHeadSimple(nn.Module):
    """A simple multi‑label head with independent linear classifiers.

    Parameters
    ----------
    feat_dim: int
        The dimensionality of the input features from the backbone.
    num_labels: int
        Number of label heads (e.g. 7 for seven artefact classes).
    num_classes: int, default=3
        Number of discrete classes per label (0, 1, 2).
    dropout: float, default=0.2
        Dropout probability applied before the classification heads.
    """
    def __init__(self, feat_dim: int, num_labels: int, num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        d = feat_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d),
        )
        self.heads = nn.ModuleList([
            nn.Linear(d, num_classes) for _ in range(num_labels)
        ])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.trunk(feats)  # (B, d)
        logits = torch.stack([h(x) for h in self.heads], dim=1)  # (B, L, C)
        return logits


class LabelTokenHead(nn.Module):
    """A multi‑label head using a learnable token per label and a lightweight transformer.

    Each label has an associated token that is modulated via FiLM by the
    global feature vector ``g`` extracted from the backbone.  A small
    Transformer encoder processes the modulated tokens, and an independent
    linear classifier predicts the class for each label.  This structure
    encourages the model to share information across labels while still
    producing separate outputs per label.

    Parameters
    ----------
    feat_dim: int
        Dimensionality of the backbone features.
    num_labels: int
        Number of labels.
    num_classes: int, default=3
        Number of classes per label.
    hidden_dim: Optional[int], default=None
        Hidden dimensionality within the head.  If ``None``, uses
        ``feat_dim``.
    n_heads: int, default=4
        Number of attention heads in the Transformer encoder.
    n_layers: int, default=1
        Number of layers in the Transformer encoder.
    mlp_ratio: float, default=2.0
        MLP expansion ratio in the Transformer encoder.
    proj_dropout: float, default=0.05
        Dropout probability applied after the FiLM modulation.
    attn_dropout: float, default=0.05
        Attention dropout in the Transformer encoder.
    ff_dropout: float, default=0.05
        Dropout applied after the Transformer encoder.
    """
    def __init__(self,
                 feat_dim: int,
                 num_labels: int,
                 num_classes: int = 3,
                 hidden_dim: Optional[int] = None,
                 n_heads: int = 4,
                 n_layers: int = 1,
                 mlp_ratio: float = 2.0,
                 proj_dropout: float = 0.05,
                 attn_dropout: float = 0.05,
                 ff_dropout: float = 0.05) -> None:
        super().__init__()
        d = hidden_dim or feat_dim
        self.num_labels = num_labels
        # Trunk to project backbone features to hidden dimension
        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, d),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.LayerNorm(d),
        )
        # Learnable tokens per label
        self.label_tokens = nn.Parameter(torch.randn(num_labels, d) * 0.02)
        # FiLM generator: takes g and outputs gamma/beta for token modulation
        self.film = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
        )
        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=int(d * mlp_ratio),
            dropout=attn_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.post = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(ff_dropout),
        )
        # Independent classifier per label
        self.heads = nn.ModuleList([
            nn.Linear(d, num_classes) for _ in range(num_labels)
        ])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, feat_dim)
        g = self.trunk(feats)  # (B, d)
        B, d = g.shape
        # Initialise label tokens for each batch
        tokens = self.label_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, L, d)
        # Generate gamma and beta via FiLM and modulate tokens
        gamma_beta = self.film(g)  # (B, 2d)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, d)
        gamma = gamma.unsqueeze(1)  # (B,1,d)
        beta  = beta.unsqueeze(1)   # (B,1,d)
        tokens = tokens * (1.0 + gamma) + beta  # FiLM modulation
        # Transformer encoder
        h = self.encoder(tokens)  # (B, L, d)
        h = self.post(h)
        # Per‑label classification
        logits = torch.stack([
            self.heads[i](h[:, i, :]) for i in range(self.num_labels)
        ], dim=1)  # (B, L, num_classes)
        return logits


class Model2DTimm(nn.Module):
    """Backbone + multi‑label head model for 2D MRI slices.

    Parameters
    ----------
    base_model: str
        Name of the timm model to use as backbone.  The backbone is
        initialised with ``num_classes=0`` so that features are returned
        instead of classification outputs.
    in_channels: int, default=1
        Number of input channels.  MRI slices are typically grayscale (1).
    num_labels: int, default=7
        Number of independent labels to predict.
    num_classes: int, default=3
        Number of classes per label (scores 0, 1, 2).
    pretrained: bool, default=True
        Whether to load pretrained weights for the backbone.
    head_type: str, default="label_tokens"
        Which head to use: either ``"simple"`` or ``"label_tokens"``.
    head_config: Optional[dict], default=None
        Additional keyword arguments passed to the head constructor.
    use_view: bool, default=False
        If ``True``, a one‑hot encoded view tensor of shape (B, 3) must be
        passed to ``forward``.  The view is embedded to the same
        dimensionality as the backbone features and added before the head.
    """
    def __init__(self,
                 base_model: str = "maxvit_tiny_tf_512.in1k",
                 in_channels: int = 1,
                 num_labels: int = 7,
                 num_classes: int = 3,
                 pretrained: bool = True,
                 head_type: str = "label_tokens",
                 head_config: Optional[Dict] = None,
                 use_view: bool = False) -> None:
        super().__init__()
        self.use_view = use_view
        # Instantiate backbone
        self.backbone = timm.create_model(
            base_model,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )
        feat_dim = self.backbone.num_features
        # Embedding layer for view if enabled
        if use_view:
            self.view_emb = nn.Linear(3, feat_dim, bias=False)
        # Select head type
        head_config = head_config or {}
        if head_type == "simple":
            self.head = MultiHeadSimple(feat_dim, num_labels, num_classes, dropout=head_config.get('dropout', 0.2))
        elif head_type == "label_tokens":
            cfg = {
                'feat_dim': feat_dim,
                'num_labels': num_labels,
                'num_classes': num_classes,
            }
            cfg.update(head_config)
            self.head = LabelTokenHead(**cfg)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x: torch.Tensor, view: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract features from backbone
        feats = self.backbone(x)  # (B, feat_dim)
        if self.use_view:
            if view is None:
                raise ValueError("Model configured with use_view=True but no view tensor provided")
            # view: (B, 3) -> embed to (B, feat_dim)
            v_emb = self.view_emb(view)
            feats = feats + v_emb
        logits = self.head(feats)  # (B, num_labels, num_classes)
        return logits
