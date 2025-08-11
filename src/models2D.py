
import torch.nn as nn
import timm

class Model2DTimmOLD(nn.Module):
    def __init__(self,
                 base_model: str = "maxvit_tiny_tf_512.in1k", #vit_tiny_patch16_224.augreg_in21k", #resnet50",
                 in_channels: int = 3,
                 num_labels: int = 7,
                 num_classes: int = 3,
                 dropout_p: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.num_labels = num_labels
        self.num_classes = num_classes

        # 🧠 Crear modelo base sin la FC original
        self.backbone = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels, num_classes=0)
        in_features = self.backbone.num_features

        # 🧱 Proyección y cabeza de clasificación
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_labels * num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)          # (B, C)
        feats = self.shared_proj(feats)   # (B, C)
        logits = self.head(feats)         # (B, 21)
        return logits.view(-1, self.num_labels, self.num_classes)  # (B, 7, 3)


import torch
import torch.nn as nn
import timm

# -------- Head 1: 7 heads independientes (simple y sólida) --------
class MultiHeadSimple(nn.Module):
    def __init__(self, feat_dim, num_labels=7, num_classes=3, head_dropout=0.2):
        super().__init__()
        d = feat_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(d),
        )
        self.heads = nn.ModuleList([nn.Linear(d, num_classes) for _ in range(num_labels)])

    def forward(self, feats):
        x = self.trunk(feats)                      # (B, d)
        logits = torch.stack([h(x) for h in self.heads], dim=1)  # (B, L, C)
        return logits

class LabelTokenHead(nn.Module):
    """
    Head multi-etiqueta con:
      - Trunk compartido suave
      - Label tokens (L, d)
      - FiLM (gamma, beta) desde g para modular tokens
      - Transformer encoder ligero
      - Heads por etiqueta (7 clasificadores independientes)
    """
    def __init__(self,
                 feat_dim, num_labels=7, num_classes=3,
                 hidden_dim=None, n_heads=4, n_layers=1,
                 mlp_ratio=2.0,
                 proj_dropout=0.05,  # ↓ menos dropout
                 attn_dropout=0.05,
                 ff_dropout=0.05):
        super().__init__()
        d = hidden_dim or feat_dim
        self.num_labels = num_labels

        # Trunk compartido
        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, d),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.LayerNorm(d),
        )

        # Tokens de etiqueta
        self.label_tokens = nn.Parameter(torch.randn(num_labels, d) * 0.02)

        # FiLM desde g: produce gamma/beta para modular tokens
        self.film = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
        )

        # Encoder ligero
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=int(d * mlp_ratio),
            dropout=attn_dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.post = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(ff_dropout),
        )

        # Heads por etiqueta
        self.heads = nn.ModuleList([nn.Linear(d, num_classes) for _ in range(num_labels)])

    def forward(self, feats):
        # feats: (B, C)
        g = self.trunk(feats)        # (B, d)
        B, d = g.shape

        # tokens base (B, L, d)
        t = self.label_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()

        # FiLM: gamma, beta desde g
        gb = self.film(g)            # (B, 2d)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, d), (B, d)
        gamma = gamma.unsqueeze(1)   # (B,1,d)
        beta  = beta.unsqueeze(1)    # (B,1,d)
        t = t * (1 + gamma) + beta   # modulación

        # Encoder
        h = self.encoder(t)          # (B, L, d)
        h = self.post(h)

        # Per-label heads
        logits = torch.stack([self.heads[i](h[:, i, :]) for i in range(self.num_labels)], dim=1)  # (B,L,3)
        return logits

# -------- Modelo principal --------
class Model2DTimm(nn.Module):
    """
    backbone timm (pretrained) -> head (simple 7-heads o label-token)
    Salida: (B, num_labels, num_classes)
    """
    def __init__(self,
                 base_model="maxvit_tiny_tf_512.in1k",
                 in_channels=1,
                 num_labels=7,
                 num_classes=3,
                 pretrained=True,
                 head_type="simple",         # "simple" | "label_tokens"
                 head_dropout=0.2,
                 dropout_p: float = 0.3,
                 drop_path_rate=0.1,         # si el backbone lo soporta
                 label_tokens_cfg=None):
        super().__init__()
        self.backbone = timm.create_model(
            base_model, pretrained=pretrained, in_chans=in_channels, num_classes=0,
            drop_path_rate=drop_path_rate
        )
        feat_dim = self.backbone.num_features

        if head_type == "simple":
            self.head = MultiHeadSimple(feat_dim, num_labels, num_classes, head_dropout=head_dropout)
        elif head_type == "label_tokens":
            cfg = dict(hidden_dim=feat_dim, n_heads=4, n_layers=3, mlp_ratio=2.0,
                       proj_dropout=head_dropout, attn_dropout=0.3, ff_dropout=0.3)
            if label_tokens_cfg:
                cfg.update(label_tokens_cfg)
            self.head = LabelTokenHead(feat_dim, num_labels, num_classes, **cfg)
        else:
            raise ValueError("head_type debe ser 'simple' o 'label_tokens'")

    def forward(self, x):
        feats = self.backbone(x)        # (B, C)
        logits = self.head(feats)       # (B, L, 3)
        return logits
