
import torch.nn as nn
import timm

class Model2DTimm(nn.Module):
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
