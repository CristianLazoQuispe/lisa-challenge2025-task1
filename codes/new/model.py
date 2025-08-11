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


class RegHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.out = nn.Linear(feat_dim // 2, 4)
        # init bias cerca del promedio del dataset (mejor que 0.5 fijo)
        # si aún no lo tienes, arranca algo razonable:
        with torch.no_grad():
            # logit(0.5)=0; w,h un poco menores (ej. 0.35)
            self.out.bias[:] = torch.tensor([0.0, 0.0, -0.62, -0.62])  # sigmoid(-0.62)~0.35

    def forward(self, feats):
        t = self.trunk(feats)               # (B, d)
        o = self.out(t)                     # (B,4)
        # parametrización estable:
        cx = torch.sigmoid(o[..., 0])
        cy = torch.sigmoid(o[..., 1])
        w  = torch.sigmoid(o[..., 2])
        h  = torch.sigmoid(o[..., 3])
        return torch.stack([cx, cy, w, h], dim=-1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def spatial_softargmax2d(attn):  # attn: (B,1,H,W) -> cx,cy en [0,1]
    B, _, H, W = attn.shape
    attn = attn.view(B, -1)
    attn = F.softmax(attn, dim=-1)
    attn = attn.view(B, 1, H, W)
    # coordenadas normalizadas [0,1]
    ys = torch.linspace(0, 1, H, device=attn.device).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.linspace(0, 1, W, device=attn.device).view(1, 1, 1, W).expand(B, 1, H, W)
    cy = (attn * ys).sum(dim=(2,3))
    cx = (attn * xs).sum(dim=(2,3))
    return cx.squeeze(1), cy.squeeze(1)

class RegHeadSpatial(nn.Module):
    """Head de bbox con información espacial.
       Predice: heatmap de centro (1 canal) + tamaño (2 canales: w,h).
       Devuelve cajas normalizadas (cx, cy, w, h) en [0,1].
    """
    def __init__(self, in_ch, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.center_head = nn.Conv2d(hidden, 1, 1)      # heatmap
        self.size_head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, 2, 1)                     # w,h (log-scale)
        )

    def forward(self, fmap):                # fmap: (B,C,Hf,Wf)
        h = self.conv(fmap)
        heat = self.center_head(h)          # (B,1,Hf,Wf)
        cx, cy = spatial_softargmax2d(heat) # [0,1], diferenciable
        sz = self.size_head(h).flatten(1)   # (B,2)
        # tamaños en (0,1) con softplus para estabilidad
        w = torch.sigmoid(sz[:, 0])
        h = torch.sigmoid(sz[:, 1])
        return torch.stack([cx, cy, w, h], dim=-1)      # (B,4)


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


        # en Model2DTimm.__init__
        #self.reg_head = nn.Sequential(
        #    nn.LayerNorm(feat_dim),
        #    nn.Linear(feat_dim, feat_dim // 2),
        #    nn.GELU(),
        #    nn.Dropout(0.05),
        #    nn.Linear(feat_dim // 2, 4),
        #    nn.Sigmoid(),  # porque el target lo normalizamos a [0,1]
        #)

        #self.reg_head = RegHead(feat_dim)
        in_ch = getattr(self.backbone, 'feature_info', None)
        in_ch = in_ch[-1]['num_chs'] if in_ch is not None else feat_dim
        self.reg_head = RegHeadSpatial(in_ch)


    def forward(self, x: torch.Tensor, view: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract features from backbone
        #feats = self.backbone(x)  # (B, feat_dim)
        #if self.use_view:
        #    if view is None:
        #        raise ValueError("Model configured with use_view=True but no view tensor provided")
        #    # view: (B, 3) -> embed to (B, feat_dim)
        #    v_emb = self.view_emb(view)
        #    feats = feats + v_emb
        #logits = self.head(feats)  # (B, num_labels, num_classes)
        #aux    = self.reg_head(feats)      # (B, 3) -> (cx, cy, r) en [0,1]
        #return logits,aux
        # fmap espacial
        fmap = self.backbone.forward_features(x)  # (B,C,Hf,Wf)
        # vector global para clasificación
        feats = self.backbone.forward_head(fmap, pre_logits=True)  # (B, feat_dim)
        if self.use_view:
            v_emb = self.view_emb(view); feats = feats + v_emb
        logits = self.head(feats)               # (B,L,3)
        bbox   = self.reg_head(fmap)            # (B,4) -> (cx,cy,w,h) normalizados
        return logits, bbox