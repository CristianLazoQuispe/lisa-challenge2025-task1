"""
Model definitions for the LISA 2025 challenge with a focus on clarity and
correctness.  These models correct several issues identified in the original
implementation:

* The ``Model2DTimm`` class can optionally incorporate the image view (axial,
  coronal or sagittal) as a one‑hot vector.  When ``use_view`` is set to
  ``True``, the view information is concatenated to the backbone features
  before passing them to the classification heads.  This allows the model to
  learn view‑specific patterns, which has been shown to improve performance
  on the LISA quality assessment task【42949289359295†L366-L374】.
* The number of input features of the head is automatically adjusted when
  ``use_view`` is enabled.

The original ``MultiHeadSimple`` and ``LabelTokenHead`` classes are kept
mostly intact, but they are documented for clarity.

Example::

    model = Model2DTimm(
        base_model="maxvit_tiny_tf_512.in1k",
        in_channels=1,
        num_labels=7,
        num_classes=3,
        pretrained=True,
        head_type="label_tokens",
        head_dropout=0.2,
        use_view=True,
    )
    # x: (B, C, H, W), view: (B, 3)
    logits = model(x, view)

"""

import torch
import torch.nn as nn
import timm


class MultiHeadSimple(nn.Module):
    """Simple multi‑head classifier.

    Takes a feature vector of size ``feat_dim`` and produces one linear head
    for each label.  Each head predicts ``num_classes`` classes.

    Args:
        feat_dim: Dimensionality of the input feature vector.
        num_labels: Number of independent labels (e.g. 7 for the LISA task).
        num_classes: Number of classes per label (3 for 0/1/2).
        head_dropout: Dropout probability applied inside the head.
    """

    def __init__(self, feat_dim: int, num_labels: int = 7,
                 num_classes: int = 3, head_dropout: float = 0.2) -> None:
        super().__init__()
        d = feat_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(d),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(d, num_classes) for _ in range(num_labels)]
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute logits for each label.

        Args:
            feats: Tensor of shape ``(B, d)``.

        Returns:
            A tensor of shape ``(B, L, C)`` containing the logits for each label.
        """
        x = self.trunk(feats)
        logits = torch.stack([h(x) for h in self.heads], dim=1)
        return logits


class LabelTokenHead(nn.Module):
    """Label‑token based head inspired by the ML‑Decoder architecture.

    This head shares a trunk across all labels, uses learnable label tokens and
    optional FiLM modulation from the global feature vector, applies a small
    Transformer encoder and finally a per‑label linear classifier.  It is
    designed to allow interaction between labels.

    Args:
        feat_dim: Dimensionality of the input feature vector.
        num_labels: Number of independent labels.
        num_classes: Number of classes per label.
        hidden_dim: Dimensionality of the hidden features (defaults to
            ``feat_dim`` if ``None``).
        n_heads: Number of attention heads in the Transformer encoder.
        n_layers: Number of encoder layers.
        mlp_ratio: Ratio of the feedforward dimension to ``hidden_dim`` in the
            Transformer encoder.
        proj_dropout: Dropout probability applied after the first linear layer.
        attn_dropout: Dropout probability for attention weights.
        ff_dropout: Dropout probability applied after the Transformer encoder.
    """

    def __init__(self,
                 feat_dim: int,
                 num_labels: int = 7,
                 num_classes: int = 3,
                 hidden_dim: int = None,
                 n_heads: int = 4,
                 n_layers: int = 1,
                 mlp_ratio: float = 2.0,
                 proj_dropout: float = 0.05,
                 attn_dropout: float = 0.05,
                 ff_dropout: float = 0.05,
                 ) -> None:
        super().__init__()
        d = hidden_dim or feat_dim
        self.num_labels = num_labels

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, d),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.LayerNorm(d),
        )

        # Label tokens
        self.label_tokens = nn.Parameter(torch.randn(num_labels, d) * 0.02)

        # FiLM from g: produce gamma/beta to modulate tokens
        self.film = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
        )

        # Light Transformer encoder
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

        # Per‑label heads
        self.heads = nn.ModuleList(
            [nn.Linear(d, num_classes) for _ in range(num_labels)]
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute logits via label tokens and Transformer encoder.

        Args:
            feats: Tensor of shape ``(B, C)``.

        Returns:
            Tensor of shape ``(B, L, C)`` with per‑label logits.
        """
        g = self.trunk(feats)  # (B, d)
        B, d = g.shape

        # base tokens (B, L, d)
        t = self.label_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()

        # FiLM: gamma, beta from g
        gb = self.film(g)  # (B, 2d)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, d), (B, d)
        gamma = gamma.unsqueeze(1)  # (B,1,d)
        beta = beta.unsqueeze(1)   # (B,1,d)
        t = t * (1 + gamma) + beta  # modulation

        # Encoder
        #print(t.shape)
        h = self.encoder(t)  # (B, L, d)
        h = self.post(h)

        # Per‑label heads
        logits = torch.stack(
            [self.heads[i](h[:, i, :]) for i in range(self.num_labels)],
            dim=1,
        )  # (B,L,3)
        return logits


class Model2DTimm(nn.Module):
    """Wrapper around timm backbones for 2D MRI quality assessment.

    This class loads a timm model, removes its classification head and
    attaches a multi‑head classifier.  It optionally concatenates a one‑hot
    representation of the view (axial/coronal/sagittal) to the backbone
    features.  When ``use_view`` is ``True``, the input dimension to the head
    increases by 3.

    Args:
        base_model: Identifier of the timm model to load.
        in_channels: Number of channels of the input image (1 for greyscale).
        num_labels: Number of independent labels (7 for the LISA task).
        num_classes: Number of classes per label (3).
        pretrained: Whether to load ImageNet weights.
        head_type: Either ``"simple"`` or ``"label_tokens"``.
        head_dropout: Dropout probability within the heads.
        dropout_p: Dropout probability in the backbone (if supported).
        drop_path_rate: Stochastic depth rate for timm models that support it.
        label_tokens_cfg: Optional dictionary of arguments for ``LabelTokenHead``.
        use_view: If ``True``, the view vector is concatenated to the features.
    """

    def __init__(self,
                 base_model: str = "maxvit_tiny_tf_512.in1k",
                 in_channels: int = 1,
                 num_labels: int = 7,
                 num_classes: int = 3,
                 pretrained: bool = True,
                 head_type: str = "label_tokens",
                 head_dropout: float = 0.2,
                 dropout_p: float = 0.3,
                 drop_path_rate: float = 0.1,
                 label_tokens_cfg: dict = None,
                 use_view: bool = False,
                 ) -> None:
        super().__init__()
        self.use_view = use_view
        # Create timm backbone without classification head
        self.backbone = timm.create_model(
            base_model,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features

        # Adjust feature dimension if using the view
        head_input_dim = feat_dim + 3 if self.use_view else feat_dim

        #print("head_input_dim:",head_input_dim)
        if head_type == "simple":
            self.head = MultiHeadSimple(
                head_input_dim, num_labels, num_classes, head_dropout=head_dropout
            )

        elif head_type == "label_tokens":
            cfg = dict(
                hidden_dim=head_input_dim,
                n_heads=5,
                n_layers=3,
                mlp_ratio=2.0,
                proj_dropout=head_dropout,
                attn_dropout=0.3,
                ff_dropout=0.3,
            )
            if label_tokens_cfg:
                cfg.update(label_tokens_cfg)
            self.head = LabelTokenHead(
                head_input_dim, num_labels, num_classes, **cfg
            )
        else:
            raise ValueError("head_type debe ser 'simple' o 'label_tokens'")

    def forward(self, x: torch.Tensor, view: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the backbone and head.

        Args:
            x: Input images of shape ``(B, C, H, W)``.
            view: Optional one‑hot vectors of shape ``(B, 3)`` indicating the
                acquisition plane.  Only used when ``use_view`` is ``True``.

        Returns:
            Logits of shape ``(B, num_labels, num_classes)``.
        """
        feats = self.backbone(x)  # (B, feat_dim)
        if self.use_view:
            assert view is not None, "view vector must be provided when use_view=True"
            if view.dim() == 1:
                view = view.unsqueeze(0)
            feats = torch.cat([feats, view], dim=-1)
        logits = self.head(feats)
        return logits